// Resolve false when cancelled so a delayed camera start cannot outlive its screen.
export function startCountdown(seconds: number, onTick: (remaining: number) => void) {
  const deadline = Date.now() + seconds * 1000;
  let finish: (completed: boolean) => void;
  let timer: ReturnType<typeof setInterval> | undefined;
  let done = false;
  const promise = new Promise<boolean>(resolve => { finish = resolve; });
  function settle(completed: boolean) {
    if (done) return;
    done = true; clearInterval(timer); finish(completed);
  }
  onTick(seconds);
  if (seconds <= 0) settle(true);
  else timer = setInterval(() => {
    const remaining = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
    onTick(remaining);
    if (!remaining) settle(true);
  }, 100);
  return { promise, cancel: () => settle(false) };
}
