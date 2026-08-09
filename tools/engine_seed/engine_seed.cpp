// Pin the engine's random number generator so a local game reproduces.
//
// `libcg` draws its entropy from `std::random_device` and exports no seed
// entry point, so `ptcg.arena.play_game(seed=...)` — which seeds Python's
// global `random` — never reached the shuffle at all. Two runs of "the same
// seed" were two different games, which is why the same frozen agent file
// scored 0.7800 / 0.7500 / 0.7367 on one cell (D66).
//
// This preload replaces `std::random_device::operator()` with a counter-based
// stream (SplitMix64) that `cabt_engine_seed` can reset, so the harness can
// open every game on a named seed and get that game back every time.
//
// LOCAL GATING ONLY. Nothing here ships: the submission never loads it, and
// on Kaggle the engine keeps its own entropy. Build with tools/engine_seed/
// build.sh and load through `ptcg.engine_seed`.

#include <cstdint>
#include <cstdlib>
#include <cstdio>

static uint64_t g_state = 0x9E3779B97F4A7C15ULL;
static uint64_t g_draws = 0;

static inline uint32_t next_draw() {
    // SplitMix64: a full-period counter mixer, so consecutive seeds give
    // unrelated streams and the harness can key a game to its index.
    g_draws++;
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (uint32_t)(z >> 32);
}

extern "C" void cabt_engine_seed(unsigned long long seed) {
    g_state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 0x243F6A8885A308D3ULL;
}

extern "C" unsigned long long cabt_engine_state(void) { return g_state; }

extern "C" unsigned long long cabt_engine_draws(void) { return g_draws; }

extern "C" int cabt_engine_seed_present(void) { return 1; }

#ifdef __APPLE__
// Two-level namespace binds libcg straight to libc++ in the shared cache, so
// DYLD_FORCE_FLAT_NAMESPACE does not redirect it; an interpose tuple does.
#define DYLD_INTERPOSE(_r, _x)                                            \
    __attribute__((used)) static struct { const void *r; const void *x; } \
    _interpose_##_x __attribute__((section("__DATA,__interpose")))        \
        = {(const void *)(unsigned long)&_r, (const void *)(unsigned long)&_x};

extern "C" unsigned real_rd_call(void *) asm("__ZNSt3__113random_deviceclEv");

extern "C" unsigned cabt_rd_call(void *self) {
    (void)self;
    return next_draw();
}
DYLD_INTERPOSE(cabt_rd_call, real_rd_call)
#else
// glibc/libstdc++: LD_PRELOAD resolves the mangled symbol directly.
extern "C" unsigned _ZNSt13random_deviceclEv(void *self) {
    (void)self;
    return next_draw();
}
extern "C" unsigned _ZNSt3__113random_deviceclEv(void *self) {
    (void)self;
    return next_draw();
}
#endif

__attribute__((constructor)) static void cabt_engine_seed_init(void) {
    const char *s = getenv("CABT_ENGINE_SEED");
    if (s) cabt_engine_seed(strtoull(s, nullptr, 10));
    if (getenv("CABT_ENGINE_SEED_VERBOSE"))
        fprintf(stderr, "[engine_seed] engine RNG pinned (seed=%s)\n",
                s ? s : "default");
}
