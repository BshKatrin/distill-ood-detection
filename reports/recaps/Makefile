LATEXMK ?= latexmk
PDFLATEX ?= pdflatex
.DEFAULT_GOAL := help
BUILD_ROOT := ../build/recaps
SUMMARY_APPENDIX_STANDALONE := \
	metrics_baseline \
	metrics_random_forest_sweep \
	metrics_perturbation_clipping_non_aggressive_cifar10 \
	metrics_perturbation_clipping_aggressive_cifar10 \
	metrics_perturbation_clipping_aggressive_resnet50 \
	metrics_perturbation_dropout_cifar10 \
	metrics_unmasked_pca_resnet18_unperturbed \
	metrics_masked_pca_fixed_unperturbed \
	metrics_masked_pca_resnet50_unperturbed \
	metrics_pixel_augmentation_resnet50_unperturbed \
	metrics_pixel_augmentation_resnet50_perturbed \
	metrics_subspace_ensemble
SUMMARY_APPENDIX_FRAGMENTS := \
	metrics_feature_denoising_pixel_augmented_embedding_cifar10_resnet18_relative_improvement \
	metrics_feature_denoising_pixel_augmented_embedding_cifar100_resnet18_relative_improvement \
	spatial_block_residual_improvement \
	spatial_block_residual_relative_improvement \
	channel_residual_improvement \
	channel_residual_relative_improvement \
	confusion_channel_replacement_improvement \
	confusion_channel_replacement_relative_improvement \
	confusion_channel_replacement_direct_improvement \
	confusion_channel_replacement_direct_relative_improvement

.PHONY: 2026-07-09 2026-07-17 2026-07-28 2026-08-17 2026-08-31 2026-08-26 clean help

2026-07-09:
	mkdir -p $(BUILD_ROOT)/2026-07-09
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-07-09) 2026-07-09/schema.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-07-09) 2026-07-09/main.tex
	cp $(BUILD_ROOT)/2026-07-09/main.pdf 2026-07-09/main.pdf

2026-07-17:
	mkdir -p $(BUILD_ROOT)/2026-07-17
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-07-17) 2026-07-17/main.tex
	cp $(BUILD_ROOT)/2026-07-17/main.pdf 2026-07-17/main.pdf

2026-07-28:
	mkdir -p $(BUILD_ROOT)/2026-07-28/appendix
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-07-28) 2026-07-28/schema.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-07-28/appendix) \
		2026-07-28/knn_exhaustive.tex
	for source in $(SUMMARY_APPENDIX_STANDALONE); do \
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
			-outdir=$(abspath $(BUILD_ROOT)/2026-07-28/appendix) \
			../outputs/latex/$$source.tex || exit 1; \
	done
	for source in $(SUMMARY_APPENDIX_FRAGMENTS); do \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error \
			-jobname=$$source \
			-output-directory=$(abspath $(BUILD_ROOT)/2026-07-28/appendix) \
			"\def\tablefile{../outputs/latex/$$source.tex}\input{2026-07-28/appendix_table.tex}" \
			|| exit 1; \
	done
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-07-28) 2026-07-28/main.tex
	cp $(BUILD_ROOT)/2026-07-28/main.pdf 2026-07-28/main.pdf

2026-08-17:
	mkdir -p $(BUILD_ROOT)/2026-08-17
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-08-17) 2026-08-17/main.tex
	cp $(BUILD_ROOT)/2026-08-17/main.pdf 2026-08-17/main.pdf

2026-08-31:
	mkdir -p $(BUILD_ROOT)/2026-08-31
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-08-31) 2026-08-31/main.tex
	cp $(BUILD_ROOT)/2026-08-31/main.pdf 2026-08-31/main.pdf

2026-08-26:
	mkdir -p $(BUILD_ROOT)/2026-08-26
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_ROOT)/2026-08-26) 2026-08-26/main.tex
	cp $(BUILD_ROOT)/2026-08-26/main.pdf 2026-08-26/main.pdf

help:
	@echo "make -C reports/recaps YYYY-MM-DD"

clean:
	rm -rf $(BUILD_ROOT)
