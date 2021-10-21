# For windows and linux
PKG = byutpl

ifeq ($(OS), Windows_NT)
	CLEANER = clean-windows
else
	CLEANER = clean-linux
endif


.PHONY: upload pybuild clean-linux clean-windows

all: build

build:
	python3 setup.py sdist bdist_wheel

upload: pybuild
	python3 -m twine upload dist/*

clean-linux: execlean-linux
	rm -f ./lpfgopt/*.so ./lpfgopt/*.dll
	rm -rf ./build ./dist ./*.egg-info

clean-windows:
	if exist .\build rmdir /S/Q .\build
	if exist .\dist  rmdir /S/Q .\dist 
	if exist .\$(PKG).egg-info rmdir /S/Q .\$(PKG).egg-info

clean: $(CLEANER)
