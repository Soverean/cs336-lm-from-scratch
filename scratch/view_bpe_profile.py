import pstats
from cs336_basics import config

profile_path = config.TS_DIR / "bpe_profile.prof"
optimize_path = config.TS_DIR / "bpe_profile_optimized.prof"
incremental_path = config.TS_DIR / "bpe_profile_incremental.prof"
indexed_path = config.TS_DIR / "bpe_profile_indexed.prof"
vocab_50Mib_path = config.TS_50_PATH / 'bpe_50MiB.prof'
optimize_pre_token_path = config.TS_50_PATH / 'bpe_50MiB_pretokens.prof'
heap_path = config.TS_50_PATH / 'bpe_50MiB_heap.prof'



print('====优化前====')
stats = pstats.Stats(str(profile_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(15)

print('====merges优化后====')
stats = pstats.Stats(str(optimize_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(15)


print('====合并优化后====')
stats = pstats.Stats(str(incremental_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(15)

print('====索引优化后====')
stats = pstats.Stats(str(indexed_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(15)

print('==============50MiB语料=============')
stats = pstats.Stats(str(vocab_50Mib_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(15)

print('==============预分词优化后===============')
stats = pstats.Stats(str(optimize_pre_token_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(15)

print('==============堆优化后===============')
stats = pstats.Stats(str(heap_path))
stats.strip_dirs().sort_stats("cumtime").print_stats(20)

# stats.sort_stats('cumtime').print_stats(15)
#
# stats.sort_stats("tottime").print_stats(15)  # 找自身计算量大的函数