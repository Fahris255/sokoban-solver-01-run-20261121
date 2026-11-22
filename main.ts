// Sokoban solver: push-optimal search, room by room.
//
// Idea in one paragraph: a Sokoban position is mostly about where the
// boxes are; the player is just "somewhere in the open region between
// boxes". So the search runs over pushes, not steps. Each state is the
// sorted box squares plus the canonical name of the player's reachable
// region (its smallest square). A* orders the frontier, guided by the
// cheapest assignment of boxes to goals under true push distances
// (computed once per level by pulling boxes backwards from the goals),
// which never overestimates, so the first solution found is still
// push-optimal. Two cheap deadlock cuts keep the frontier honest: a
// box resting on a square from which no goal is reachable by pushes
// can never finish, and a solid 2x2 of walls/boxes with a box off goal
// is frozen forever. Levels built as a hallway with separate rooms are
// split at the hallway: each room is solved on its own (boxes never
// leave their room, a push out of the door is a push onto a dead
// square), and the room optima add up to the level optimum. Anything
// that does not split is searched whole. All budgets count expansions,
// never the clock, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order keeps every tie broken the same way each run.
const DIRS = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
] as const;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever the text says: nothing leaves.
  for (let x = 0; x < w; x++) { wall[x] = 1; wall[(h - 1) * w + x] = 1; }
  for (let y = 0; y < h; y++) { wall[y * w] = 1; wall[y * w + w - 1] = 1; }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Push distances: dist[goalIndex][square] = fewest pushes that move a
// lone box from `square` onto that goal, ignoring the other boxes, or
// -1 when impossible. Built backwards from each goal: a box now on `c`
// could just have been pushed there from `c - d`, with the player
// standing on `c - 2d`; both squares must be open floor.
function pushDistances(lv: Level): Int32Array[] {
  const { w, wall, goals } = lv;
  const size = wall.length;
  return goals.map((goal) => {
    const dist = new Int32Array(size).fill(-1);
    dist[goal] = 0;
    const queue = [goal];
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi]!;
      for (const { dx, dy } of DIRS) {
        const d = dy * w + dx;
        const from = c - d;
        const stand = c - 2 * d;
        if (from < 0 || stand < 0 || from >= size || stand >= size) continue;
        if (wall[from] || wall[stand] || dist[from] !== -1) continue;
        dist[from] = dist[c]! + 1;
        queue.push(from);
      }
    }
    return dist;
  });
}

// Cheapest box-to-goal assignment for a box set, by trying every
// permutation of goals (at most a handful of boxes per search here).
// Returns Infinity when some box cannot reach any free goal at all.
function assignmentCost(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const cost: number[][] = boxes.map((b) => dists.map((d) => d[b]!));
  let best = Infinity;
  const used = new Array<boolean>(dists.length).fill(false);
  const walk = (i: number, acc: number): void => {
    if (acc >= best) return;
    if (i === n) { best = acc; return; }
    for (let g = 0; g < dists.length; g++) {
      if (used[g]) continue;
      const c = cost[i]![g]!;
      if (c < 0) continue;
      used[g] = true;
      walk(i + 1, acc + c);
      used[g] = false;
    }
  };
  walk(0, 0);
  return best;
}

// Flood the player's reachable squares around `occupied` boxes.
// Fills `seen` with `stamp` and returns the region's smallest square.
function flood(lv: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = lv;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const nb = [p - w, p + w, p - 1, p + 1];
    for (let k = 0; k < 4; k++) {
      const q = nb[k]!;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Is the box just pushed onto `t` part of a solid 2x2 block (walls and
// boxes only) that includes a box off its goal? Such a block never
// moves again, so the position is dead.
function frozenSquare(lv: Level, occupied: Uint8Array, isGoal: Uint8Array, t: number): boolean {
  const { w, wall } = lv;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const cells = [a, a + 1, a + w, a + w + 1];
    let solid = true;
    let bad = false;
    for (const c of cells) {
      if (!(wall[c] || occupied[c])) { solid = false; break; }
      if (occupied[c] && !isGoal[c]) bad = true;
    }
    if (solid && bad) return true;
  }
  return false;
}

// Shortest walking route (no pushes) from `from` to `to`, or '' when
// already there. Used only when turning a push sequence into moves.
function walkPath(lv: Level, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const { w, wall } = lv;
  const prev = new Int32Array(wall.length).fill(-1);
  const letter = new Array<string>(wall.length).fill('');
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const { dx, dy, walk } of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      letter[q] = walk;
      queue.push(q);
    }
  }
  let out = '';
  for (let p = to; p !== from; p = prev[p]!) out = letter[p] + out;
  return out;
}

interface PushStep { from: number; dir: number }

// A* over pushes for one (sub-)level. Returns the push list, or null
// when the expansion budget runs out or no solution exists.
function solvePushes(lv: Level, maxExpansions: number): PushStep[] | null {
  const { w, wall, goals } = lv;
  const size = wall.length;
  const isGoal = new Uint8Array(size);
  for (const g of goals) isGoal[g] = 1;
  const dists = pushDistances(lv);
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) live[i] = dists.some((d) => d[i]! >= 0) ? 1 : 0;
  if (lv.boxes.some((b) => !live[b])) return null;
  if (lv.boxes.every((b) => isGoal[b])) return [];

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  interface Node {
    boxes: number[];
    player: number; // square the player stands on (after the push)
    parent: number;
    step: PushStep | null;
    g: number;
  }
  const nodes: Node[] = [];
  const bestG = new Map<string, number>();
  const keyOf = (boxes: number[], regionLow: number): string =>
    regionLow + '|' + boxes.join(',');

  // Binary heap ordered by f, then by deeper g (closer to the goal on
  // ties), then by insertion order: fully deterministic.
  const heap: { f: number; g: number; idx: number }[] = [];
  const before = (a: { f: number; g: number; idx: number }, b: { f: number; g: number; idx: number }): boolean =>
    a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.idx < b.idx)));
  const heapPush = (e: { f: number; g: number; idx: number }): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): { f: number; g: number; idx: number } => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1]!, heap[c]!)) c++;
        if (!before(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  for (const b of lv.boxes) occupied[b] = 1;
  const startRegion = flood(lv, occupied, lv.player, seen, ++stamp, stack);
  for (const b of lv.boxes) occupied[b] = 0;
  const h0 = assignmentCost(lv.boxes, dists);
  if (!Number.isFinite(h0)) return null;
  nodes.push({ boxes: lv.boxes.slice(), player: lv.player, parent: -1, step: null, g: 0 });
  bestG.set(keyOf(lv.boxes, startRegion), 0);
  heapPush({ f: h0, g: 0, idx: 0 });

  let expansions = 0;
  while (heap.length > 0) {
    const { idx } = heapPop();
    const node = nodes[idx]!;
    const boxes = node.boxes;
    if (boxes.every((b) => isGoal[b])) {
      const steps: PushStep[] = [];
      for (let t = idx; nodes[t]!.step !== null; t = nodes[t]!.parent) steps.push(nodes[t]!.step!);
      steps.reverse();
      return steps;
    }
    if (++expansions > maxExpansions) return null;
    for (const b of boxes) occupied[b] = 1;
    flood(lv, occupied, node.player, seen, ++stamp, stack);
    const regionStamp = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const { dx, dy } = DIRS[di]!;
        const d = dy * w + dx;
        const target = b + d;
        const stand = b - d;
        if (seen[stand] !== regionStamp) continue; // player cannot get behind it
        if (wall[target] || occupied[target] || !live[target]) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenSquare(lv, occupied, isGoal, target);
        let regionLow = -1;
        let nextBoxes: number[] | null = null;
        if (!frozen) {
          nextBoxes = boxes.slice();
          nextBoxes[bi] = target;
          nextBoxes.sort((a, b2) => a - b2);
          regionLow = flood(lv, occupied, b, mark, ++stamp, stack);
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen || nextBoxes === null) continue;
        const g = node.g + 1;
        const key = keyOf(nextBoxes, regionLow);
        const known = bestG.get(key);
        if (known !== undefined && known <= g) continue;
        const h = assignmentCost(nextBoxes, dists);
        if (!Number.isFinite(h)) continue;
        bestG.set(key, g);
        const childIdx = nodes.length;
        nodes.push({ boxes: nextBoxes, player: b, parent: idx, step: { from: b, dir: di }, g });
        heapPush({ f: g + h, g, idx: childIdx });
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

// Turn a push list into the full move string, walking the player to
// the back of each box first. Replays the boxes as it goes.
function renderMoves(lv: Level, steps: PushStep[]): string {
  const occupied = new Uint8Array(lv.wall.length);
  for (const b of lv.boxes) occupied[b] = 1;
  let player = lv.player;
  let moves = '';
  for (const { from, dir } of steps) {
    const { dx, dy, push } = DIRS[dir]!;
    const d = dy * lv.w + dx;
    moves += walkPath(lv, occupied, player, from - d);
    moves += push;
    occupied[from] = 0;
    occupied[from + d] = 1;
    player = from;
  }
  return moves;
}

// Split a level at its hallway: squares on the player's row that no
// box can ever use (dead for pushes) are cut away; each remaining
// component holding boxes is one room. Returns null when the shape
// does not decompose cleanly (box in the hallway, goals cut off, or a
// component whose box and goal counts differ).
function splitRooms(lv: Level): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const dists = pushDistances(lv);
  const live = new Uint8Array(wall.length);
  for (let i = 0; i < wall.length; i++) live[i] = dists.some((d) => d[i]! >= 0) ? 1 : 0;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      const nb = [p - w, p + w, p - 1, p + 1];
      for (const q of nb) {
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length > 0 && r.boxes.length !== r.goals.length)) return null;
  const withBoxes = rooms.filter((r) => r.boxes.length > 0);
  return withBoxes.length > 0 ? withBoxes : null;
}

const ROOM_BUDGET = 1_200_000;
const WHOLE_BUDGET = 1_500_000;

function solveLevel(text: string): string {
  const lv = parseLevel(text);
  if (lv.boxes.length === 0) return '';
  const rooms = splitRooms(lv);
  if (rooms !== null && rooms.length > 1 || (rooms !== null && rooms.length === 1)) {
    // Solve room by room. Other rooms' boxes stand as walls; a room
    // already finished stands on its goals. The player walks on from
    // wherever the previous room's solution left them.
    let player = lv.player;
    let allMoves = '';
    let ok = true;
    const finished = new Set<number>();
    for (const room of rooms) {
      const wall = lv.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.boxes) if (!finished.has(b)) wall[b] = 1;
        for (const g of other.goals) if (other.boxes.every((b) => finished.has(b))) wall[g] = 1;
      }
      const sub: Level = {
        w: lv.w, h: lv.h, wall,
        goals: room.goals.slice(),
        boxes: room.boxes.slice(),
        player,
      };
      const steps = solvePushes(sub, ROOM_BUDGET);
      if (steps === null) { ok = false; break; }
      const moves = renderMoves(sub, steps);
      allMoves += moves;
      // Advance the player and mark this room's boxes as done.
      let p = player;
      for (const ch of moves.toLowerCase()) p += ch === 'u' ? -lv.w : ch === 'd' ? lv.w : ch === 'l' ? -1 : 1;
      player = p;
      for (const b of room.boxes) finished.add(b);
    }
    if (ok) return allMoves;
  }
  const steps = solvePushes(lv, WHOLE_BUDGET);
  return steps === null ? '' : renderMoves(lv, steps);
}

for (const id of readdirSync(INPUT).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
