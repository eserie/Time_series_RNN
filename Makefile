SRC = $(shell find ./ -type f -name "*.py")
TARGETS = $(shell find ./ -type f -name "*.ipynb")
IPYKERNEL_NAME = online_portfolio2
PACKAGE_NAME = ./

.PHONY: all
all: run-notebooks

.PHONY: sync
sync:
	jupytext --sync -- $(SRC)

.PHONY: run-notebooks
run-notebooks: sync
	for f in $(TARGETS); do jupytext --execute $${f}; done

.PHONY: set-kernel
set-kernel:
	jupytext --set-kernel $(IPYKERNEL_NAME) -- $(SRC)
	$(MAKE) sync

.PHONY: autoflake
autoflake: 
	autoflake --in-place --remove-all-unused-imports -r $(PACKAGE_NAME)

.PHONY: isort
isort:
	python -m isort --profile black --treat-comment-as-code "# %%" $(PACKAGE_NAME)
	# python -m isort --profile black --float-to-top $(PACKAGE_NAME)

.PHONY: black
black:
	python -m black $(PACKAGE_NAME)

.PHONY: format
format: autoflake isort black
