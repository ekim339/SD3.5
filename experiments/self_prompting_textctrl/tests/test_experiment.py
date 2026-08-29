import tempfile, unittest
from pathlib import Path
from PIL import Image
from experiments.self_prompting_textctrl.data import expand_jobs, prepare_samples
from experiments.self_prompting_textctrl.metrics import text_metrics


class ExperimentTests(unittest.TestCase):
    def test_metrics_are_case_sensitive(self):
        self.assertEqual(text_metrics("ABCDE","abcde")["ACC"],0)
        self.assertEqual(text_metrics("ABCDE","abcde",case_sensitive=False)["ACC"],1)

    def test_target_families_and_job_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); shard=root/"shard"
            (shard/"i_s").mkdir(parents=True); (shard/"mask_s").mkdir()
            Image.new("RGB",(16,16),"gray").save(shard/"i_s/a.png")
            Image.new("L",(16,16),255).save(shard/"mask_s/a.png")
            (shard/"i_s.txt").write_text("a.png abcde\n",encoding="utf-8")
            config={"sample_count":1,"sample_seed":1,"target_seed":2,"generation_seed":3,
                    "dataset":{"roots":[str(shard)],"source_labels":"i_s.txt",
                               "source_dir":"i_s","mask_dir":"mask_s"},
                    "input_noise":{"seed":4,"minimum_standard_deviation":1,
                                   "maximum_standard_deviation":1},
                    "targets":{"letters":"abcXYZ","special_characters":"!?"}}
            output=root/"output"; samples=prepare_samples(config,root,output)
            jobs=expand_jobs(samples,config,output)
            self.assertEqual(len(jobs),16)
            self.assertEqual(samples[0]["case_upper"],samples[0]["case_lower"].upper())
            expected_lengths=(5,5,5,3,5,5)
            self.assertEqual(tuple(len(samples[0][f"special_{i}"]) for i in range(1,7)),expected_lengths)
            paired={(job["model"],job["target_key"]):job["target_text"] for job in jobs}
            for key in ("case_upper","case_lower",*(f"special_{i}" for i in range(1,7))):
                self.assertEqual(paired[("regular",key)],paired[("self_prompting",key)])

    def test_self_prompting_only_jobs(self):
        sample={"sample_index":0,"source_text":"abcde","case_upper":"ABCDE",
                "case_lower":"abcde",**{f"special_{i}":"!!!!!" for i in range(1,7)}}
        jobs=expand_jobs([sample],{"generation_seed":3},Path("/tmp/output"),
                         models=("self_prompting",))
        self.assertEqual(len(jobs),8)
        self.assertEqual({job["model"] for job in jobs},{"self_prompting"})


if __name__=="__main__": unittest.main()
