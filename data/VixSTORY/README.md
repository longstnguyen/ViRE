# VixSTORY

[↑ Back to datasets](../../README.md#datasets) · [Paper](https://doi.org/10.1007/978-981-92-2014-4_19) · [Citation](#citation)

Structured multihop question answering for university admission advising across institutions. We share this dataset as a bonus resource from our related work, alongside ViRE's ten-dataset retrieval benchmark.

## Files

| File | Contents |
| :-- | :-- |
| [`contexts.csv`](contexts.csv) | Source passages with `context_id`, `title`, and `text` |
| [`qa.csv`](qa.csv) | Questions, answers, hop levels, and relevant passage IDs in `gold_context_ids` |

Use `gold_context_ids` in the question records to associate each question with its relevant `context_id` values in the passage corpus. The `split` column in `qa.csv` identifies the provided data splits.

## Citation

If you use **VixSTORY**, please cite our corresponding paper:

Long S. T. Nguyen, Tin T. Ngo, Dung N. H. Le, Quynh T. N. Vo, Dung H. Nguyen, Khang N. Le, and Tho T. Quan, “[A Benchmark for Structured Multihop Reasoning in Cross-Institution University Admission Advisory Question Answering](https://doi.org/10.1007/978-981-92-2014-4_19)”, in _Trends and Applications in Knowledge Discovery and Data Mining_, Springer Nature Singapore, pages 235–247, 2027.

```bibtex
@inproceedings{10.1007/978-981-92-2014-4_19,
    title        = {{A Benchmark for Structured Multihop Reasoning in Cross-Institution
                   University Admission Advisory Question Answering}},
    author       = {Nguyen, Long S. T. and Ngo, Tin T. and Le, Dung N. H. and Vo, Quynh T. N.
                   and Nguyen, Dung H. and Le, Khang N. and Quan, Tho T.},
    year         = 2027,
    booktitle    = {Trends and Applications in Knowledge Discovery and Data Mining},
    publisher    = {Springer Nature Singapore},
    address      = {Singapore},
    pages        = {235--247},
    isbn         = {978-981-92-2014-4},
    editor       = {Long, Cheng and Shi, Jieming and Liang, Yuxuan and Huang, Xiao and Choi,
                   Byron and Song, Yuanfeng and Yiu, Man Lung},
}
```
