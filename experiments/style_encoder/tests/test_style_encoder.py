from experiments.style_encoder.data import expand_jobs, patch_tokens
from experiments.style_encoder.metrics import text_metrics


def test_patch_token_indices_are_row_major():
    assert patch_tokens(0, 0) == [0, 1, 2, 3, 16, 17, 18, 19,
                                  32, 33, 34, 35, 48, 49, 50, 51]
    assert patch_tokens(3, 3) == [204, 205, 206, 207, 220, 221, 222, 223,
                                  236, 237, 238, 239, 252, 253, 254, 255]
    assert sorted(token for row in range(4) for col in range(4)
                  for token in patch_tokens(row, col)) == list(range(256))


def test_expand_jobs_has_five_proportions_and_seventeen_patch_conditions(tmp_path):
    sample = {"sample_index": 0, "target_text": "abc12", "source_text": "hello"}
    jobs = expand_jobs([sample], {"seed": 7,
                                  "masking_proportions": [0, .1, .3, .5, .7]}, tmp_path)
    assert len(jobs) == 22
    assert sum(job["experiment"] == "proportion" for job in jobs) == 5
    assert sum(job["experiment"] == "patch" for job in jobs) == 17


def test_metrics():
    assert text_metrics("abcde", "abcde") == {"ACC": 1.0, "NED": 1.0, "CER": 0.0}
    assert text_metrics("abcde", "abXde") == {"ACC": 0.0, "NED": 0.8, "CER": 0.2}
