# ClimateChangeFromMachineLearningResearch

This is a very messy repo containing code for experiments and the generating of graphs for our paper. Please open an issue if you have any questions!

```
@misc{henderson2020systematic,
    title={Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning},
    author={Peter Henderson and Jieru Hu and Joshua Romoff and Emma Brunskill and Dan Jurafsky and Joelle Pineau},
    year={2020},
    eprint={2002.05651},
    archivePrefix={arXiv},
    primaryClass={cs.CY}
}
```

Script to run translation experiments:

```
paper_specific/translate.py
```

Script to run computer vision experiments:

```
paper_specific/run_inference.py
```

For the bar plot of transfomer vs conv:

```
python paper_specific/bar_plot_word.py experiment_results/translation/ --experiment_set_names "Conv (3-8 words)" "Transformer (3-8 words)" "Conv (25-30 words)" "Transformer (25-30 words)" "Conv (3-50 words)" "Transformer (3-50 words)" "Conv (45-50 words)" "Transformer (45-50 words)" --experiment_set_filters "conv.wmt14.en-fr_*_min3_max8" "transformer.wmt14.en-fr_*_min3_max8" "conv.wmt14.en-fr_*_min25_max30" "transformer.wmt14.en-fr_*_min25_max30" "conv.wmt14.en-fr_*_min3_max50" "transformer.wmt14.en-fr_*_min3_max50" "conv.wmt14.en-fr_*_min45_max50" "transformer.wmt14.en-fr_*_min45_max50" --y_axis_var total_power
```

For the region and carbon emissions line plot:

```
python paper_specific/region_lineplot.py
```

To randomly sample NeurIPS papers:

```
python paper_specific/parse_neurips_papers.py https://papers.nips.cc/book/advances-in-neural-information-processing-systems-32-2019 --n 100 --link-filters paper
```

To compare estimation methods:

```
python paper_specific/bar_plot_estimation_methods.py experiment_results/rl/ --experiment_set_names "Pong PPO" --experiment_set_filters "Pong*ppo2"
```

To generate site appendices:

```
create-compute-appendix ./experiment_results/ --site_spec appendix_website_definitions/website_defs.json --output_dir ./docs/
```
