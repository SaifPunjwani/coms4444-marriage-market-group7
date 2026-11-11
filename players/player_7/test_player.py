"""Tests for Player 7: the recognition toggle, trust checks and shared matching."""

import uuid

import pytest

from core.engine import Engine
from models.player import Advertisement, DateContext, GameContext, PlayerSnapshot
from players.greedy_player import GreedyPlayer
from players.player_7 import player as module
from players.player_7.compare import most_positive
from players.player_7.player import Player7, _enlarge, _match, _plain_ad, _signed_ad
from players.random_player import RandomPlayer


class On(Player7):
	recognise = True


class Off(Player7):
	recognise = False


def make(cls, profile=(5, 5, 5, 5, 5), ideal=(5, 5, 5, 5, 5), index=0, people=20, rounds=100):
	snapshot = PlayerSnapshot(uuid.UUID(int=index + 1), index, profile, ideal)
	return cls(snapshot, GameContext(people, 5, 2, rounds, 100.0, 18))


def meet(me, who, shown, stated, ad, round_number=1, number=1):
	date = DateContext(round_number, me.rounds - round_number + 1, who, ad, number, False)
	return me.decide(date, shown, stated)


def run(players, bonus=100, seed=1, rounds=100):
	return Engine(
		players=players, rounds=rounds, bonus=bonus, seed=seed, timeout=0, keep_records=False
	).run()


# ----------------------------------------------------------------------
# The toggle
# ----------------------------------------------------------------------


def test_ad_follows_the_toggle():
	profile, ideal = (3, 7, 4, 9, 1), (6, 2, 0, 5, 8)
	assert make(On, profile, ideal).advertise() == _signed_ad(profile, ideal)
	plain = make(Off, profile, ideal).advertise()
	assert plain == _plain_ad(profile, ideal, 2)
	# Best monotonic trait, and the ideal furthest from the middle.
	assert (plain.own_dim, plain.own_value) == (1, 7)
	assert (plain.seek_dim, plain.seek_value) == (2, 0)


def test_only_signed_people_are_allies_when_recognising():
	profile, ideal = (3, 7, 4, 9, 1), (6, 2, 0, 5, 8)
	signed, plain = _signed_ad(profile, ideal), _plain_ad(profile, ideal, 2)
	assert signed != plain
	me = make(On)
	meet(me, 1, profile, ideal, signed)
	meet(me, 2, profile, ideal, plain)
	assert me._ally(1)
	assert not me._ally(2)


def test_every_honest_person_is_an_ally_when_not_recognising():
	profile, ideal = (3, 7, 4, 9, 1), (6, 2, 0, 5, 8)
	me = make(Off)
	meet(me, 1, profile, ideal, _signed_ad(profile, ideal))
	meet(me, 2, profile, ideal, _plain_ad(profile, ideal, 2))
	assert me._ally(1)
	assert me._ally(2)


def test_nothing_reads_the_signature_when_not_recognising(monkeypatch):
	"""With recognition off, a whole market plays without the signature ever
	being computed, so no copy can be told from any other honest person."""

	def forbidden(*args):
		raise AssertionError('the signature was used with recognition off')

	monkeypatch.setattr(module, '_signed_ad', forbidden)
	result = run([Off] * 10 + [GreedyPlayer] * 6 + [RandomPlayer] * 4, rounds=40)
	assert result['faults'] == []
	assert result['married_people'] > 0


# ----------------------------------------------------------------------
# Trust
# ----------------------------------------------------------------------


def test_an_ideal_that_contradicts_their_own_ad_is_not_believed():
	me = make(Off, profile=(5, 5, 5, 5, 5))
	asked_for_nine = Advertisement(own_dim=0, own_value=6, seek_dim=3, seek_value=9)
	meet(me, 1, (6, 6, 5, 5, 5), (5, 5, 5, 5, 5), asked_for_nine)
	meet(me, 2, (6, 6, 5, 5, 5), (5, 5, 5, 9, 5), asked_for_nine)
	assert not me._candid(1)
	assert not me._trusted(1)
	assert me._trusted(2)
	# Flattery is priced as an average ideal, not as the perfect fit it claims.
	assert me._with_me(1)[1] < me._with_me(2)[1]


def test_someone_whose_true_ideal_we_are_is_still_trusted():
	me = make(Off, profile=(5, 5, 5, 5, 5))
	ad = Advertisement(own_dim=0, own_value=6, seek_dim=3, seek_value=5)
	meet(me, 1, (6, 6, 5, 5, 5), (5, 5, 5, 5, 5), ad)
	assert me._trusted(1)


def test_a_changed_profile_or_ideal_is_not_trusted():
	me = make(Off)
	ad = Advertisement(own_dim=0, own_value=6, seek_dim=3, seek_value=5)
	meet(me, 1, (6, 6, 5, 5, 5), (5, 5, 5, 5, 5), ad)
	meet(me, 1, (6, 8, 5, 5, 5), (5, 5, 5, 5, 5), ad, round_number=2, number=2)
	meet(me, 2, (6, 6, 5, 5, 5), (5, 5, 5, 5, 5), ad)
	meet(me, 2, (6, 6, 5, 5, 5), (5, 5, 4, 5, 5), ad, round_number=2, number=2)
	assert not me._steady(1)
	assert not me._candid(2)
	assert not me._trusted(1)
	assert not me._trusted(2)


def test_an_inflated_ad_only_costs_the_trait_it_inflated():
	"""Their ad claimed an 8, their profile shows a 6: we believe the 6, which
	is worse for us, and take every other trait as shown."""
	me = make(Off, profile=(5, 5, 5, 5, 5), ideal=(9, 5, 5, 5, 5))
	boasted = Advertisement(own_dim=0, own_value=8, seek_dim=3, seek_value=5)
	meet(me, 1, (6, 5, 5, 5, 5), (5, 5, 5, 5, 5), boasted)
	ours, their = me._with_me(1)
	assert ours == 100 - 9
	assert their == 100 - 9 / 2
	assert me._trusted(1)


def test_masking_counts_twos_and_eights_on_plain_traits_only():
	me = make(Off, ideal=(0, 0, 0, 5, 10))
	ad = Advertisement(own_dim=0, own_value=8, seek_dim=3, seek_value=5)
	meet(me, 1, (8, 8, 8, 8, 8), (5, 5, 5, 5, 5), ad)
	# Trait 2: an 8 hiding a 10 costs 100 instead of 64. Trait 3: 25 instead
	# of 9. Trait 4: nothing beyond 8 is worse than 8 when we want a 10.
	assert me._masking(1) == (100 - 64) + (25 - 9)


# ----------------------------------------------------------------------
# The shared matching
# ----------------------------------------------------------------------


def graph(pairs):
	edges, weight = {}, {}
	for a, b, total in pairs:
		edges.setdefault(a, set()).add(b)
		edges.setdefault(b, set()).add(a)
		weight[(min(a, b), max(a, b))] = total
	return edges, weight


def test_matching_finds_the_most_couples():
	# A path of four: taking the heavy middle edge would strand both ends.
	mate = _match(*graph([(0, 1, 10), (1, 2, 90), (2, 3, 10)]))
	assert mate == {0: 1, 1: 0, 2: 3, 3: 2}


def test_enlarging_works_through_an_odd_cycle():
	# A ring of five with a tail on three of its corners and one chord. Starting
	# from two couples inside the ring, everyone can still be paired, but only
	# by re-pairing around the ring.
	ring = [(0, 1, 5), (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 0, 5), (1, 3, 5)]
	edges, _ = graph(ring + [(0, 5, 5), (2, 6, 5), (4, 7, 5)])
	near = [sorted(edges[p]) for p in sorted(edges)]
	match = [1, 0, 3, 2, -1, -1, -1, -1]
	_enlarge(near, match)
	assert all(partner != -1 for partner in match)
	assert all(match[partner] == person for person, partner in enumerate(match))
	assert all(partner in near[person] for person, partner in enumerate(match))


def test_matching_prefers_happier_couples_without_dropping_anyone():
	# Two ways to make two couples; one is worth far more.
	mate = _match(*graph([(0, 1, 10), (2, 3, 10), (0, 2, 80), (1, 3, 80)]))
	assert mate == {0: 2, 2: 0, 1: 3, 3: 1}
	# A leftover person is taken in place of a worse partner.
	mate = _match(*graph([(0, 1, 10), (0, 2, 70)]))
	assert mate == {0: 2, 2: 0}


def test_schedule_pairs_everyone_exactly_once():
	people = [make(Off, index=i, people=8) for i in range(8)]
	met = set()
	for round_number in range(1, 8):
		for person in people:
			partner = person._scheduled(round_number)
			assert partner != person.index
			assert people[partner]._scheduled(round_number) == person.index
			met.add(frozenset((person.index, partner)))
	assert len(met) == 8 * 7 // 2


@pytest.mark.parametrize('cls', [On, Off])
@pytest.mark.parametrize('bonus', [50, 100])
def test_a_market_of_copies_marries_as_many_as_possible(cls, bonus):
	for seed in (1, 2, 3):
		result = run([cls] * 20, bonus=bonus, seed=seed)
		assert result['faults'] == []
		positive = sum(p['positive'] for p in result['players'])
		assert positive == most_positive(result['players'], bonus, 2)


@pytest.mark.parametrize('cls', [On, Off])
def test_plays_a_mixed_market_without_faults(cls):
	result = run([cls] * 8 + [GreedyPlayer] * 8 + [RandomPlayer] * 4, rounds=60)
	assert result['faults'] == []
	assert any(p['married'] for p in result['players'][:8])
