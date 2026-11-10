from model_maxxing.evaluation.runner import token_budget_chunks


def _check(lengths, bs, new, budget):
    ch = token_budget_chunks(lengths, bs, new, budget)
    assert [i for i, _ in ch] == [0] + [j for _, j in ch[:-1]] and ch[-1][1] == len(lengths)   # partición contigua
    for i, j in ch:
        assert j - i <= bs
        if j - i > 1:
            assert (j - i) * (max(lengths[i:j]) + new) <= budget
    return ch


def test_sin_presupuesto_es_tamano_fijo():
    assert token_budget_chunks(list(range(10)), 4, 512, None) == [(0, 4), (4, 8), (8, 10)]


def test_presupuesto_parte_los_lotes_largos():
    lengths = [90] * 20 + [700, 800, 3703]
    ch = _check(lengths, 16, 512, 16384)
    assert ch[0] == (0, 16)                      # los cortos siguen en lotes de 16
    assert ch[-1] == (21, 23) or ch[-1] == (22, 23)
    assert (22, 23) in ch or any(i <= 22 < j and (j - i) * (3703 + 512) <= 16384 for i, j in ch)


def test_ejemplo_mayor_que_el_presupuesto_va_solo():
    assert _check([100, 50000], 16, 512, 16384) == [(0, 1), (1, 2)]
