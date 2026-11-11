"""Group 7: honest dating, then marriage by a shared matching rule.

If both people on a date are honest, the date reveals the other person's true
profile and true ideal, so each of us can work out exactly how happy *both*
people would be in that marriage. After dating most of the market, a person
therefore knows the whole compatibility graph among the singles they have met:
an edge joins two people when both would end up with positive happiness.

Our primary goal is the share of people with positive happiness, which is the
size of a matching in that graph. Every copy of this player computes the same
deterministic matching from the same information and pursues only its assigned
partner, so copies agree on who marries whom without sharing any state. The
matching repeatedly pairs the person with the fewest compatible partners, so
people with many options do not take the only option of someone with few.

Who joins the shared matching depends on RECOGNISE. Turned off, everyone who
has been honest with us joins it, our ad is an ordinary one, and nothing in
this file tells a copy of this player from any other honest person. Turned on,
our ad's dimensions are a checksum of our true profile and ideal, which a date
reveals, and only people whose ad carries that signature join. Anyone left out
will not wait for a matching they do not compute, so we take them once they
beat both our assigned partner and a patience bar: what we expect to end up
with if we hold out for the people we can still meet. The bar is reasoned dimension by dimension. Under the values
we have seen, each dimension has an expected cost and a variance, and since
dimensions are drawn independently these add up to the mean and spread of our
happiness with a random partner. The bar falls to zero as the unmet people
run out, so we stay picky early and never miss the end. We stop counting on
anyone who snubs us twice, and we block outsiders who are hopeless for us so
the engine stops spending our dates on them.

1. Explore: list people we have not met, ranked by how well their ad fits us.
2. Commit: once we have met everyone, list only our assigned partner, ask for
   repeat dates, and propose from the second date on.
3. Fall back: if the assigned partner does not respond, or time is running
   out, propose to any compatible person, best first.

We never propose unless our own happiness would be positive. A partner whose
projected profile changes between dates or contradicts their ad is treated as
no better than a random person and is left out of the shared graph. A partner
whose stated ideal is just our own profile is flattering us: we assume an
average ideal for them and that the value our ad asked for was nudged by 2.
"""

import math
from statistics import NormalDist

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	DateOutcome,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement

VALUES = range(11)
NORMAL = NormalDist()

# Whether to single out copies of this player. Off: the shared matching is open
# to every honest person and nothing here identifies a copy. On: our ad is a
# signature, and only people who carry it join the matching.
RECOGNISE = False

# Pseudo-counts per value before any date: a broad bell around 5, so the first
# ad-based estimates are sensible and real observations soon dominate.
PRIOR = [0.5 + 1.5 * math.exp(-((v - 5) ** 2) / 12.5) for v in VALUES]

# Rounds without a date with our assigned partner before we set them aside,
# and how long they stay set aside.
PATIENCE = 4
COOLDOWN = 10
# Rounds without meeting anyone new before we stop exploring: whoever is left
# has probably marked us incompatible, so we can never be paired with them.
STALL = 5
# In the last few rounds, propose to anyone compatible.
ENDGAME = 6
# The share of the market that must still be single for us to keep holding
# out for the shared matching rather than take early offers.
WAITING = 0.5
# Spare rounds, beyond one per person, for meeting everyone in a small market
# before everybody commits on the same round.
SLACK = 4
# A person not seen among the candidates for this many rounds has married.
GONE_AFTER = 3
# Unanswered proposals or refused repeat dates before we stop counting on
# someone outside our group: they keep dating us but will not marry us.
SNUBS = 2
# The chance that any one person we have yet to meet turns into a real option
# for us, when deciding how picky to be with outsiders.
CHANCE = 0.2
# How far someone who flatters us may have nudged the value our ad asked for.
NUDGE = 2


def _plain_ad(profile: Profile, ideal: Profile, monotonic: int) -> Advertisement:
	"""An ordinary honest ad: the monotonic trait we score best on, and the
	dimension where our ideal is furthest from the middle of the scale."""
	showable = range(monotonic) if monotonic else range(len(profile))
	own = max(showable, key=lambda dim: profile[dim])
	seek = max(range(len(ideal)), key=lambda dim: abs(ideal[dim] - 5))
	return Advertisement(own_dim=own, own_value=profile[own], seek_dim=seek, seek_value=ideal[seek])


def _signed_ad(profile: Profile, ideal: Profile) -> Advertisement:
	"""An honest ad with both dimensions picked by a checksum of our profile
	and ideal. A date shows us a partner's true profile and ideal, and if they
	produce exactly the ad that partner showed, the partner runs this player.
	Other groups pick dimensions by value, so they rarely match by chance."""
	d = len(profile)
	h = sum(
		(i + 1) * (3 * p + 7 * q + 1) for i, (p, q) in enumerate(zip(profile, ideal, strict=True))
	)
	own, seek = h % d, h // d % d
	return Advertisement(own_dim=own, own_value=profile[own], seek_dim=seek, seek_value=ideal[seek])


def _enlarge(near: list[list[int]], match: list[int]) -> None:
	"""Grow a matching until no larger one exists (Edmonds' blossom algorithm).
	``near[i]`` lists who person ``i`` can pair with and ``match[i]`` is their
	partner or -1. People already matched stay matched, perhaps to someone
	else."""
	n = len(near)

	def augment(root: int) -> bool:
		parent, base = [-1] * n, list(range(n))
		seen = [False] * n
		seen[root] = True
		queue = [root]

		def common(a: int, b: int) -> int:
			marked = [False] * n
			while True:
				a = base[a]
				marked[a] = True
				if match[a] == -1:
					break
				a = parent[match[a]]
			while True:
				b = base[b]
				if marked[b]:
					return b
				b = parent[match[b]]

		def contract(v: int, top: int, child: int, inside: list[bool]) -> None:
			while base[v] != top:
				inside[base[v]] = inside[base[match[v]]] = True
				parent[v] = child
				child = match[v]
				v = parent[match[v]]

		for v in queue:
			for u in near[v]:
				if base[v] == base[u] or match[v] == u:
					continue
				if u == root or (match[u] != -1 and parent[match[u]] != -1):
					top = common(v, u)
					inside = [False] * n
					contract(v, top, u, inside)
					contract(u, top, v, inside)
					for i in range(n):
						if inside[base[i]]:
							base[i] = top
							if not seen[i]:
								seen[i] = True
								queue.append(i)
				elif parent[u] == -1:
					parent[u] = v
					if match[u] == -1:
						while u != -1:
							before = parent[u]
							after = match[before]
							match[u], match[before] = before, u
							u = after
						return True
					seen[match[u]] = True
					queue.append(match[u])
		return False

	for person in range(n):
		if match[person] == -1:
			augment(person)


def _match(edges: dict[int, set[int]], weight: dict[tuple[int, int], float]) -> dict[int, int]:
	"""Pair people up in a compatibility graph: first as many couples as
	possible, then as much total happiness as we can give them. Returns each
	matched person's partner. Every step is deterministic, so everyone who
	starts from the same graph arrives at the same couples.

	It starts with whoever has the fewest options taking their best partner,
	enlarges that to a largest possible matching, and then repeats two
	improvements that keep the number of couples: someone trades their partner
	for a leftover person they are happier with, or two couples swap partners
	for a higher total. Each raises the total, so the search always ends."""
	people = sorted(edges)
	spot = {p: i for i, p in enumerate(people)}
	near = [sorted(spot[q] for q in edges[p]) for p in people]
	fits = [set(row) for row in near]
	n = len(people)

	def worth(a: int, b: int) -> float:
		x, y = people[a], people[b]
		return weight[(x, y) if x < y else (y, x)]

	match = [-1] * n
	left = [set(row) for row in near]
	while True:
		live = [i for i in range(n) if left[i]]
		if not live:
			break
		u = min(live, key=lambda i: (len(left[i]), i))
		v = min(left[u], key=lambda i: (-worth(u, i), i))
		match[u], match[v] = v, u
		for x in (u, v):
			for y in left[x]:
				left[y].discard(x)
			left[x] = set()
	_enlarge(near, match)

	improved = True
	while improved:
		improved = False
		for u in range(n):
			if match[u] != -1:
				continue
			# Every neighbour of a leftover person is matched, or the matching
			# would not be the largest.
			for a in near[u]:
				if worth(a, u) > worth(a, match[a]):
					match[match[a]] = -1
					match[a], match[u] = u, a
					improved = True
					break
		for a in range(n):
			for c in range(a + 1, n):
				b, d = match[a], match[c]
				if b < a or d < c:
					continue
				best, now = None, worth(a, b) + worth(c, d)
				for (p, q), (r, t) in (((a, c), (b, d)), ((a, d), (b, c))):
					if q in fits[p] and t in fits[r] and worth(p, q) + worth(r, t) > now:
						best, now = ((p, q), (r, t)), worth(p, q) + worth(r, t)
				if best:
					for p, q in best:
						match[p], match[q] = q, p
					improved = True
	return {people[i]: people[j] for i, j in enumerate(match) if j != -1}


class Player7(BasePlayer):
	"""One honest person who marries the partner a shared matching assigns."""

	recognise = RECOGNISE

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.counts = [list(PRIOR) for _ in range(self.dims)]
		self.costs: tuple[list[float], list[float]] | None = None
		# What each person showed and said on every date with us.
		self.shown: dict[int, list[Profile]] = {}
		self.stated: dict[int, list[Profile]] = {}
		self.ads: dict[int, Advertisement] = {}
		self.last_seen: dict[int, int] = {}
		self.proposed_to_me: set[int] = set()
		# Partners who ignored us, and how long we have waited on the current one.
		self.unresponsive: dict[int, int] = {}
		# How often each person ignored our proposal or our request to meet again.
		self.snubs: dict[int, int] = {}
		self.asked: dict[int, Decision] = {}
		self.last_new = 0
		# In a market small enough to meet everyone, the round after which we
		# stop exploring and commit. It depends only on public numbers, so
		# everyone who plays this way switches on the same round and plans
		# from the same knowledge.
		meet_all = self.people - 1 + SLACK
		self.commit_after = meet_all if meet_all + ENDGAME < self.rounds else None
		self.exploring = True
		# Whether most of the market has already married: then few people are
		# left to plan with, and waiting for the shared matching costs more
		# than it gains.
		self.hasty = False
		self.target: int | None = None
		self.waited = 0
		self.round = 0
		self.unmet = 0

	# ------------------------------------------------------------------
	# Scoring
	# ------------------------------------------------------------------

	def _cost(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _expected_costs(self) -> tuple[list[float], list[float]]:
		"""Per dimension, our expected cost from a random partner's value, and a
		random partner's expected cost from ours, under the learned distribution."""
		if self.costs is None:
			mine, theirs = [], []
			for dim, row in enumerate(self.counts):
				total = sum(row)
				want, have = self.ideal[dim], self.profile[dim]
				mine.append(
					sum(n * self._cost(dim, want, v) for v, n in zip(VALUES, row, strict=True))
					/ total
				)
				theirs.append(
					sum(n * self._cost(dim, w, have) for w, n in zip(VALUES, row, strict=True))
					/ total
				)
			self.costs = (mine, theirs)
		return self.costs

	def _from_ad(self, ad: Advertisement) -> float:
		"""Our expected happiness with someone we have not met, from their ad."""
		mine, theirs = self._expected_costs()
		ours = sum(mine) - mine[ad.own_dim]
		ours += self._cost(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		their = sum(theirs) - theirs[ad.seek_dim]
		their += self._cost(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return self.bonus - ours - their / 2

	def _steady(self, who: int) -> bool:
		"""They have shown us the same profile on every date."""
		shown = self.shown[who]
		return all(p == shown[0] for p in shown)

	def _candid(self, who: int) -> bool:
		"""Their stated ideal has never changed and agrees with what their own
		ad asked for. Ads are posted before anyone meets, so an ideal made up
		to flatter us afterwards nearly always contradicts the ad."""
		stated = self.stated[who]
		if any(s != stated[0] for s in stated):
			return False
		ad = self.ads.get(who)
		return ad is None or stated[0][ad.seek_dim] == ad.seek_value

	def _trusted(self, who: int) -> bool:
		"""Nothing they have shown or said has contradicted itself."""
		return self._steady(who) and self._candid(who)

	def _with_me(self, who: int) -> tuple[float, float]:
		"""(our happiness, their happiness) if we married ``who``, doubting only
		what they have given us reason to doubt."""
		mine, theirs = self._expected_costs()
		steady, ad = self._steady(who), self.ads.get(who)
		ours = 0.0
		for dim in range(self.dims):
			cost = max(self._cost(dim, self.ideal[dim], p[dim]) for p in self.shown[who])
			if not steady:
				# A profile that changes is worth no more than a stranger's.
				cost = max(cost, mine[dim])
			elif ad is not None and dim == ad.own_dim:
				# Their ad and their profile may disagree on this one trait:
				# believe whichever is worse for us and take the rest as shown.
				cost = max(cost, self._cost(dim, self.ideal[dim], ad.own_value))
			ours += cost
		their = max(decrement(s, self.profile, self.monotonic) for s in self.stated[who])
		if not self._candid(who):
			# Their stated ideal tells us nothing, so assume no better than an
			# average one, and that whoever tailors an ideal to us may also
			# have nudged the trait our ad asked for.
			their = max(their, sum(theirs))
			seek = self.advertise().seek_dim
			for p in self.shown[who]:
				near = range(max(0, p[seek] - NUDGE), min(10, p[seek] + NUDGE) + 1)
				nudged = max(self._cost(seek, self.ideal[seek], v) for v in near)
				ours += max(0, nudged - self._cost(seek, self.ideal[seek], p[seek]))
				break
		return self.bonus - ours - their / 2, self.bonus - their - ours / 2

	def _masking(self, who: int) -> float:
		"""How much worse for us ``who`` would be if every 2 or 8 they show on
		a non-monotonic trait were really something more extreme. Showing a
		value two steps in from the end of the scale is how a profile hides a
		0 or a 10, and nothing about one person can tell the two apart."""
		shown, extra = self.shown[who][0], 0.0
		for dim in range(self.monotonic, self.dims):
			beyond = {2: (0, 1), 8: (9, 10)}.get(shown[dim], ())
			worst = max((self._cost(dim, self.ideal[dim], v) for v in beyond), default=0)
			extra += max(0, worst - self._cost(dim, self.ideal[dim], shown[dim]))
		return extra

	def _optimistic(self, who: int) -> float:
		"""Our happiness with ``who`` if what they showed and said were true."""
		ours = min(
			sum(self._cost(d, self.ideal[d], p[d]) for d in range(self.dims))
			for p in self.shown[who]
		)
		their = min(decrement(s, self.profile, self.monotonic) for s in self.stated[who])
		return self.bonus - ours - their / 2

	def _pair(self, a: int, b: int) -> float | None:
		"""Total happiness if two other people married, or None unless both
		would be positive. Uses what each showed us, which honest people share."""
		ta, da = self.shown[a][0], self.stated[a][0]
		tb, db = self.shown[b][0], self.stated[b][0]
		ca = decrement(da, tb, self.monotonic)
		cb = decrement(db, ta, self.monotonic)
		ha, hb = self.bonus - ca - cb / 2, self.bonus - cb - ca / 2
		return ha + hb if ha > 0 and hb > 0 else None

	# ------------------------------------------------------------------
	# Matching
	# ------------------------------------------------------------------

	def _ally(self, who: int) -> bool:
		"""Whether ``who`` joins the shared matching: they have been honest with
		us and, when we single out copies, their ad is exactly the signed one
		we would write for the profile and ideal they showed us."""
		if not self._trusted(who):
			return False
		if not self.recognise:
			return True
		return self.ads.get(who) == _signed_ad(self.shown[who][0], self.stated[who][0])

	def _singles(self) -> list[int]:
		return [
			who
			for who in self.shown
			if self.round - self.last_seen.get(who, -GONE_AFTER) < GONE_AFTER and self._ally(who)
		]

	def _assign(self) -> int | None:
		"""Our partner in the matching of every single we know, or None.

		Everyone who plays this way and knows the same people builds the same
		graph and gets the same answer, so assignments agree."""
		me = self.index
		people = sorted(self._singles())
		edges: dict[int, set[int]] = {p: set() for p in people + [me]}
		weight: dict[tuple[int, int], float] = {}
		for p in people:
			if self.round - self.unresponsive.get(p, -COOLDOWN) < COOLDOWN or self._snubbed(p):
				continue
			ours, their = self._with_me(p)
			if ours > 0 and their > 0:
				edges[me].add(p)
				edges[p].add(me)
				weight[(min(me, p), max(me, p))] = ours + their
		for i, a in enumerate(people):
			for b in people[i + 1 :]:
				total = self._pair(a, b)
				if total is not None:
					edges[a].add(b)
					edges[b].add(a)
					weight[(min(a, b), max(a, b))] = total

		return _match(edges, weight).get(me)

	def _market(self) -> tuple[float, float]:
		"""Mean and spread of our happiness with a random partner.

		Each dimension is reasoned about on its own. Under the values seen so
		far it has an expected cost and a variance, once for what we would lose
		on it and once for the half of a random partner's loss that we share.
		Dimensions are drawn independently, so the totals are just the sums."""
		mean, variance = self.bonus, 0.0
		for dim, row in enumerate(self.counts):
			total = sum(row)
			ours = [self._cost(dim, self.ideal[dim], v) for v in VALUES]
			theirs = [self._cost(dim, w, self.profile[dim]) / 2 for w in VALUES]
			for costs in (ours, theirs):
				first = sum(n * c for n, c in zip(row, costs, strict=True)) / total
				second = sum(n * c * c for n, c in zip(row, costs, strict=True)) / total
				mean -= first
				variance += second - first * first
		return mean, math.sqrt(variance)

	def _bar(self, rounds_left: int) -> float:
		"""The happiness an outsider must offer us now: what we expect to end up
		with if we hold out for the people we can still meet.

		Holding out with nobody left to meet is worth zero. Each further person
		is a random partner who turns into a real option with probability
		CHANCE, and raises the bar by the amount they are expected to beat it.
		It falls to zero as the unmet people run out."""
		mean, spread = self._market()
		bar = 0.0
		if spread <= 0:
			return bar
		for _ in range(min(self.unmet, max(0, rounds_left - ENDGAME))):
			z = (mean - bar) / spread
			gain = CHANCE * spread * (NORMAL.pdf(z) + z * NORMAL.cdf(z))
			if gain < 0.01:
				break
			bar += gain
		return bar

	def _snubbed(self, who: int) -> bool:
		if who in self.proposed_to_me or self._ally(who):
			return False
		return self.snubs.get(who, 0) >= SNUBS

	def _best_compatible(self) -> list[int]:
		"""Singles we would be positive with, best for us first. Outsiders count
		even when we think they would not be: whether to accept is their call."""
		scored = []
		for who in self.shown:
			if self.round - self.last_seen.get(who, -GONE_AFTER) >= GONE_AFTER:
				continue
			ours, their = self._with_me(who)
			likely = their > 0 or who in self.proposed_to_me
			if ours > 0 and (likely or not self._ally(who)):
				scored.append((not self._snubbed(who), likely, ours, who))
		return [who for *_, who in sorted(scored, reverse=True)]

	# ------------------------------------------------------------------
	# Interface
	# ------------------------------------------------------------------

	def advertise(self) -> Advertisement:
		if self.recognise:
			return _signed_ad(self.profile, self.ideal)
		return _plain_ad(self.profile, self.ideal, self.monotonic)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.round = turn.round
		for c in candidates:
			self.ads[c.index] = c.ad
			self.last_seen[c.index] = turn.round
		fresh = [c for c in candidates if c.index not in self.shown]
		self.unmet = len(fresh)
		self.hasty = turn.singles < WAITING * self.people
		limit = self.list_limit if self.list_limit is not None else len(candidates)

		late = turn.rounds_left <= ENDGAME
		if self.commit_after is not None:
			exploring = turn.round <= self.commit_after
		else:
			# Too many people to meet them all from the start: keep meeting new
			# ones only once there is time to meet everyone who is left.
			exploring = (
				bool(fresh)
				and turn.round - self.last_new <= STALL
				and turn.rounds_left > len(fresh) + ENDGAME
			)
		self.exploring = exploring
		self._retarget()

		offered = {c.index for c in candidates}
		target = self.target if self.target in offered else None
		if target is not None and not exploring:
			return [target]
		if exploring and self.commit_after is not None:
			# Meet everyone once. The engine pairs us with anyone on our list
			# who also listed us, in no particular order, so the only way to
			# keep to the shared schedule is to name nobody else.
			due = self._scheduled(turn.round)
			if any(c.index == due for c in fresh):
				return [due]
			# They are gone or we have met: any new face, best-fitting ads
			# kept if the list has to be cut.
			fresh.sort(key=lambda c: (-self._from_ad(c.ad), c.index))
			chosen = [c.index for c in fresh]
			if not chosen and target is not None:
				chosen.append(target)
			return chosen[:limit]
		chosen = [target] if target is not None else []
		later = []
		if late or not exploring:
			for w in self._best_compatible():
				if w in offered and w not in chosen:
					# Someone who keeps turning us down goes behind new faces.
					(later if self._snubbed(w) else chosen).append(w)
		fresh.sort(key=lambda c: (-self._from_ad(c.ad), c.index))
		chosen += [c.index for c in fresh] + later
		return chosen[:limit]

	def _scheduled(self, round_number: int) -> int:
		"""Who a round-robin schedule pairs us with this round. Anyone can work
		it out from N and their own index, and over N-1 rounds it pairs every
		two people exactly once, so nobody spends a round waiting."""
		last = self.people - 1
		turn = (round_number - 1) % last
		if self.index == last:
			return turn
		partner = (2 * turn - self.index) % last
		return last if partner == self.index else partner

	def _retarget(self) -> None:
		target = self._assign()
		if target != self.target:
			self.target, self.waited = target, 0
		elif target is not None and self.commit_after is not None and self.exploring:
			# Before the shared commit round nobody is expected to respond.
			self.waited = 0
		elif target is not None:
			self.waited += 1
			if self.waited > PATIENCE:
				self.unresponsive[target] = self.round
				self.target, self.waited = self._assign(), 0

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		who = date.partner
		self.round = date.round
		self.ads[who] = date.partner_ad
		self.last_seen[who] = date.round
		if who not in self.shown:
			self.last_new = date.round
			for dim, (v, w) in enumerate(zip(observed, stated_ideal, strict=True)):
				self.counts[dim][v] += 1
				self.counts[dim][w] += 1
			self.costs = None
		self.shown.setdefault(who, []).append(tuple(observed))
		self.stated.setdefault(who, []).append(tuple(stated_ideal))

		decision = self._decide(date, who)
		self.asked[who] = decision
		return decision

	def _decide(self, date: DateContext, who: int) -> Decision:
		ours, their = self._with_me(who)
		if ours <= 0:
			# An outsider who is hopeless for us even taken at face value gets
			# blocked, so the engine stops spending our dates on them. Copies
			# are never blocked: they must stay visible for the shared matching.
			if self._optimistic(who) <= 0 and not self._ally(who):
				return Decision(incompatible=True)
			return Decision()

		ally = self._ally(who)
		# In a market too big to meet everyone there is always someone else to
		# try, so we do not stake a marriage on a value that may be masked.
		# Near the end, a doubtful match still beats none.
		roomy = self.commit_after is None and date.rounds_left > ENDGAME
		if roomy and not (ally and self.recognise) and ours <= self._masking(who):
			return Decision()
		# Whether they will hold off until everyone has met everyone, as the
		# shared matching needs. A signed copy always does. With recognition
		# off we cannot know who does, so we assume it of every honest person
		# until most of the market has married without waiting.
		waits = ally and (self.recognise or not self.hasty)
		willing = their > 0 or who in self.proposed_to_me or not ally
		bar = self._bar(date.rounds_left)
		if who == self.target:
			self.waited = 0
			if self.exploring and (waits or ours < bar):
				return Decision()
			return Decision(again=True, propose=date.date_number >= 2)

		# Someone who already proposed, anyone compatible once the matching has
		# nobody for us, or anyone compatible near the end.
		late = date.rounds_left <= ENDGAME or self.target is None and not self.exploring
		# Someone who will not wait for the matching: take them if they beat
		# both the partner it offers us and what the rest of the market is
		# likely to offer. Not while we are still meeting everyone, though,
		# unless the market is emptying around us: every early marriage takes
		# a partner away from somebody the matching could have placed.
		if not waits and willing and (self.hasty or not self.exploring):
			mine = self._with_me(self.target)[0] if self.target in self.shown else 0
			if ours >= max(mine, bar):
				return Decision(again=True, propose=date.date_number >= 2)
		if late or who in self.proposed_to_me and ours >= bar:
			return Decision(again=willing, propose=willing and date.date_number >= 2)
		return Decision()

	def on_outcome(self, outcome: DateOutcome) -> None:
		who = outcome.partner
		if outcome.partner_proposed:
			self.proposed_to_me.add(who)
		asked = self.asked.pop(who, None)
		snubbed = asked is not None and (
			asked.propose and not outcome.partner_proposed or asked.again and not outcome.again
		)
		if snubbed and not outcome.married:
			self.snubs[who] = self.snubs.get(who, 0) + 1
