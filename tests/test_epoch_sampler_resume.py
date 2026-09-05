from crackmeanflow.common.data import EpochRandomSampler

def test_epoch_random_sampler_is_no_replacement_and_epoch_deterministic():
    s=EpochRandomSampler(100,seed=17)
    s.set_epoch(3); a=list(iter(s))
    s.set_epoch(3); b=list(iter(s))
    assert a==b
    assert sorted(a)==list(range(100))
    s.set_epoch(4); c=list(iter(s))
    assert c!=a and sorted(c)==list(range(100))
