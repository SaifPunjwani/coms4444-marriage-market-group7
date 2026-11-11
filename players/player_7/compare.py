"""Compare versions of this player on the same seeds and the same seats.

    uv run players/player_7/compare.py                 recognition on against off
    uv run players/player_7/compare.py --base main     also the player committed on a git ref
    uv run players/player_7/compare.py --big           add 1000-person markets (slower)
    uv run players/player_7/compare.py --seeds 20      more seeds (default 10)

Every version plays the seats this group would hold in the end-of-project
format (configs/tournament.json): 20 people, 100 rounds, a bonus of 50, 100
and 200, in a market of its own copies ("solo"), a half-and-half market with
each other group ("pairs") and one market with every group ("mix"). ``--big``
adds markets of 100 people per group over 150 rounds, and one of 1000 copies.

Reported for each version, over the seats it held: the share that ended with
positive happiness (the primary goal), mean happiness (the tie-break), how
many marriages ended at or below zero, and faults. For markets of its own
copies it also shows the most people any set of marriages could have left
positive, so falling short of that is the player's own doing.

Versions are named ``on`` and ``off`` (the working tree with RECOGNISE set
either way) or by a git ref, which loads the player as committed there.
"""

import argparse
import subprocess
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.distributions import distribution_from_settings  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402
from models.profile import decrement  # noqa: E402
from players.player_7.player import _enlarge  # noqa: E402

OURS = '7'
BONUSES = (50, 100, 200)
# The per-dimension map staff used for the 1000-person class runs.
CLASS_MAP = '0:normal,1:uniform,2:normal,3:uniform,4:10'


@cache
def groups() -> dict:
	"""Every group's player that loads, keyed by its code, in numeric order."""
	loaded, _ = discover()
	return {code: loaded[code] for code in sorted((c for c in loaded if c.isdigit()), key=int)}


@cache
def build(version: str) -> type:
	"""The player class for a version: ``on``, ``off`` or a git ref."""
	if version in ('on', 'off'):
		from players.player_7.player import Player7

		return type(f'Player7_{version}', (Player7,), {'recognise': version == 'on'})
	shown = subprocess.run(
		['git', 'show', f'{version}:players/player_7/player.py'],
		cwd=ROOT,
		capture_output=True,
		text=True,
		check=True,
	)
	module = types.ModuleType(f'player_7_at_{version}')
	exec(compile(shown.stdout, f'{version}:player.py', 'exec'), module.__dict__)
	return module.Player7


def most_positive(people: list[dict], bonus: float, monotonic: int) -> int:
	"""The most people any set of marriages could leave with positive happiness:
	twice a maximum matching of the pairs where both would be positive."""
	n = len(people)

	def happy(a: int, b: int) -> bool:
		mine = decrement(people[a]['ideal'], people[b]['profile'], monotonic)
		theirs = decrement(people[b]['ideal'], people[a]['profile'], monotonic)
		return bonus - mine - theirs / 2 > 0 and bonus - theirs - mine / 2 > 0

	near = [[b for b in range(n) if b != a and happy(a, b)] for a in range(n)]
	match = [-1] * n
	_enlarge(near, match)
	return sum(1 for partner in match if partner != -1)


def play(job: tuple) -> tuple:
	version, arena, roster, settings = job
	settings = dict(settings)
	distribution, label = 'mixed', None
	if settings.pop('class_map', False):
		distribution, label = distribution_from_settings(
			distribution='mixed', dims=5, monotonic=2, dim_dist=CLASS_MAP
		)
	mine, others = build(version), groups()
	result = Engine(
		players=[mine if code == OURS else others[code] for code in roster],
		dims=5,
		monotonic=2,
		distribution=distribution,
		distribution_name=label,
		keep_records=False,
		**settings,
	).run()
	seats = [p for p, code in zip(result['players'], roster, strict=True) if code == OURS]
	ceiling = None
	if all(code == OURS for code in roster):
		ceiling = most_positive(result['players'], settings['bonus'], 2)
	return (
		version,
		arena,
		settings['bonus'],
		len(seats),
		sum(p['positive'] for p in seats),
		sum(p['happiness'] for p in seats),
		sum(1 for p in seats if p['married'] and p['happiness'] <= 0),
		len(result['faults']),
		ceiling,
	)


def schedule(versions: list[str], seeds: list[int], big: bool) -> tuple[list, list]:
	codes = list(groups())
	rivals = [code for code in codes if code != OURS]
	small, large = [], []
	for version in versions:
		for seed in seeds:
			for bonus in BONUSES:
				settings = {'rounds': 100, 'bonus': bonus, 'seed': seed}
				small.append((version, 'solo', [OURS] * 20, settings))
				for rival in rivals:
					first, second = sorted((OURS, rival), key=codes.index)
					roster = [first] * 10 + [second] * 10
					small.append((version, 'pairs', roster, settings))
				mix = [codes[i % len(codes)] for i in range(20)]
				small.append((version, 'mix', mix, settings))
		if big:
			settings = {'rounds': 150, 'bonus': 100, 'class_map': True}
			for seed in seeds[:4]:
				roster = [code for code in codes for _ in range(100)]
				large.append((version, 'big mix', roster, settings | {'seed': seed}))
			large.append((version, 'big solo', [OURS] * 1000, settings | {'seed': seeds[0]}))
	return small, large


def report(rows: list[tuple], versions: list[str]) -> None:
	def line(title: str, keep) -> None:
		cells = []
		for version in versions:
			mine = [r for r in rows if r[0] == version and keep(r)]
			seats = sum(r[3] for r in mine)
			if not seats:
				return
			positive, happiness = sum(r[4] for r in mine), sum(r[5] for r in mine)
			negative, faults = sum(r[6] for r in mine), sum(r[7] for r in mine)
			cell = f'{100 * positive / seats:5.1f}% {happiness / seats:6.1f} {negative:3d}'
			if faults:
				cell += f' !{faults}'
			ceilings = [r[8] for r in mine]
			if all(c is not None for c in ceilings):
				cell += f' (max {100 * sum(ceilings) / seats:5.1f}%)'
			cells.append(cell)
		print(f'{title:<22}' + '   '.join(f'{c:<34}' for c in cells))

	print(f'{"":<22}' + '   '.join(f'{v:<34}' for v in versions))
	print(f'{"":<22}' + '   '.join(f'{"positive   mean  negative":<34}' for _ in versions))
	small = ('solo', 'pairs', 'mix')
	line('tournament format', lambda r: r[1] in small)
	for bonus in BONUSES:
		line(f'  B={bonus}', lambda r, b=bonus: r[1] in small and r[2] == b)
	for arena in small:
		line(f'  {arena}', lambda r, a=arena: r[1] == a)
		for bonus in BONUSES:
			line(f'    {arena} B={bonus}', lambda r, a=arena, b=bonus: r[1] == a and r[2] == b)
	line('1000 people, mixed', lambda r: r[1] == 'big mix')
	line('1000 people, solo', lambda r: r[1] == 'big solo')


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('versions', nargs='*', help='on, off, or a git ref (default: on off)')
	parser.add_argument('--base', action='append', default=[], help='a git ref to include')
	parser.add_argument('--seeds', type=int, default=10, help='seeds 1..N (default 10)')
	parser.add_argument('--big', action='store_true', help='add 1000-person markets')
	parser.add_argument('--jobs', type=int, default=6, help='parallel processes (default 6)')
	parser.add_argument('--timeout', type=float, default=1.0, help='seconds per player call')
	args = parser.parse_args()

	versions = [*args.base, *(args.versions or ['on', 'off'])]
	small, large = schedule(versions, list(range(1, args.seeds + 1)), args.big)
	for jobs in (small, large):
		for job in jobs:
			job[3]['timeout'] = args.timeout
	with ProcessPoolExecutor(args.jobs) as pool:
		rows = list(pool.map(play, small, chunksize=4))
	if large:
		# Fewer at once: these are heavy enough to trip each other's timeouts.
		with ProcessPoolExecutor(min(args.jobs, 4)) as pool:
			rows += list(pool.map(play, large))
	report(rows, versions)


if __name__ == '__main__':
	main()
