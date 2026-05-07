// Build-time flag for the standalone (queue-less) DEMON deployment. The
// production daydream webapp leaves this unset and the queue-admit flow
// populates useSessionStore.wsUrl before any /api/* fetch fires; the
// dev/local server has no queue, so we punch a hole through the
// "wait for wsUrl" gates with this flag. Cleaning up the queue
// scaffolding entirely is a separate pass — see Hunter's note about
// "queue code boundaries later".
export const LOCAL_MODE = process.env.NEXT_PUBLIC_LOCAL_MODE === "1";
