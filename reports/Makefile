LATEXMK ?= latexmk
BUILD_ROOT := build
FINAL_BUILD := $(abspath $(BUILD_ROOT)/final-report)

.DEFAULT_GOAL := help
.PHONY: help final-report recap clean

help:
	@echo "make -C reports final-report"
	@echo "make -C reports recap DATE=YYYY-MM-DD"
	@echo "make -C reports clean (scratch files only)"

final-report:
	mkdir -p $(FINAL_BUILD)
	cd final-report && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(FINAL_BUILD) main.tex
	cp $(FINAL_BUILD)/main.pdf final-report/main.pdf

recap:
	@test -n "$(DATE)" || { echo "Set DATE=YYYY-MM-DD" >&2; exit 1; }
	$(MAKE) -C recaps "$(DATE)"

clean:
	rm -rf $(BUILD_ROOT)
