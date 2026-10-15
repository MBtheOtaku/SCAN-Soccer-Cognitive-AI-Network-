import type { GoalPoint } from "./analysis";

export function nudgeGoalCorner(points: GoalPoint[], index: number, dx: number, dy: number, width: number, height: number): GoalPoint[] {
  if (!points[index] || ![dx, dy, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return points;
  return points.map((point, i): GoalPoint => i === index
    ? [Math.max(0, Math.min(1, point[0] + dx / width)), Math.max(0, Math.min(1, point[1] + dy / height))]
    : point);
}

export function validGoalCorners(points: GoalPoint[]): boolean {
  if (points.length !== 4 || points.some((point) => point.length !== 2 || point.some((v) => !Number.isFinite(v) || v < 0 || v > 1))) return false;
  let area = 0;
  for (let i = 0; i < 4; i++) {
    const a = points[i], b = points[(i + 1) % 4], c = points[(i + 2) % 4];
    if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0) return false;
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area / 2 >= .001 && points[0][0] < points[1][0] && points[3][0] < points[2][0] && points[0][1] < points[3][1] && points[1][1] < points[2][1];
}
