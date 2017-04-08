BUILD := build

all: cuda main

main:
	make -f $(BUILD)/makefile.main

cuda:
	make -f $(BUILD)/makefile.cuda

clean:
	make -f $(BUILD)/makefile.cuda clean
	make -f $(BUILD)/makefile.main clean
	
clobber: clean