.PHONY: docs tests help

define verify_install
	if ! [ "$$(pip list | grep $(1) -c)" = "1" ]; then\
	  if [ $(1) = "fontai" ]; then\
	    pip install -e src/fontai/fontai ;\
	  else\
	    pip install $(1);\
	  fi;\
	fi
endef

docs: ## Updates documentation
	@$(call verify_install, pdoc3);\
	rm -rf docs/* && pdoc --html -c latex_math=True -o docs src/fontai/fontai && mv docs/fontai/* docs/ && rm -rf docs/fontai;\

tests: ## Tests package
	@$(call verify_install, pytest);
	@pytest src/fontai/tests/

help: ## Shows Makefile's help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)