<p align="center"><img src="../../docs/assets/usage-header.svg" width="100%" alt="ViRE datasets and evaluation guide"></p>

# ViHERMES

[↑ Back to datasets](../../README.md#datasets) · [Paper](https://doi.org/10.1007/978-981-92-0071-9_22) · [Citation](#citation)

Multihop question answering over Vietnamese healthcare regulations. We share this dataset as a bonus resource from our related work, alongside ViRE's ten-dataset retrieval benchmark.

## Files

| File | Contents |
| :-- | :-- |
| [`contexts.csv`](contexts.csv) | Source passages with `context_id`, `title`, and `text` |
| [`qa.csv`](qa.csv) | Questions, answers, hop levels, and relevant passage IDs in `gold_context_ids` |

Use `gold_context_ids` in the question records to associate each question with its relevant `context_id` values in the passage corpus.

## Citation

If you use **ViHERMES**, please cite our corresponding paper:

Long S. T. Nguyen, Quan M. Bui, Tin T. Ngo, Quynh T. N. Vo, Dung N. H. Le, and Tho T. Quan, “[ViHERMES: A Graph-Grounded Multihop Question Answering Benchmark and System for Vietnamese Healthcare Regulations](https://doi.org/10.1007/978-981-92-0071-9_22)”, in _Intelligent Information and Database Systems_, Springer Nature Singapore, pages 321–336, 2026.

```bibtex
@inproceedings{nguyen2026-aciids-vihermes,
    title        = {{ViHERMES: A Graph-Grounded Multihop Question Answering Benchmark and
                   System for Vietnamese Healthcare Regulations}},
    author       = {Nguyen, Long S. T. and Bui, Quan M. and Ngo, Tin T. and Vo, Quynh T. N.
                   and Le, Dung N. H. and Quan, Tho T.},
    editor       = {Nguyen, Ngoc Thanh and Chen, Chun-Hao and Fujita, Hamido and Hong,
                   Tzung-Pei and Manolopoulos, Yannis and Wojtkiewicz, Krystian},
    year         = 2026,
    booktitle    = {Intelligent Information and Database Systems},
    publisher    = {Springer Nature Singapore},
    address      = {Singapore},
    pages        = {321--336},
    doi          = {10.1007/978-981-92-0071-9_22},
    isbn         = {978-981-92-0071-9},
    url          = {https://link.springer.com/chapter/10.1007/978-981-92-0071-9_22},
}
```
