# use it as:

.PHONY: clean all

clean: 
	rm -f *.pdf

supp_quiescent.pdf:
	python3 plot_quiescent.py 

synthetic_figure.pdf:
	python3 plot_synthetic.py 

schematic_figure.pdf:
	python3 plot_schematic.py 

primate_figure.pdf:
	python3 plot_primate.py 

supp_dim_figure.pdf:
	python3 plot_dim_supp.py 

supp_RT_figure.pdf:
	python3 plot_RT_supp.py

supp_primate_figure.pdf:
	python3 plot_primate_supp.py 


all: clean synthetic_figure.pdf schematic_figure.pdf primate_figure.pdf supp_dim_figure.pdf supp_RT_figure.pdf supp_primate_figure.pdf supp_quiescent.pdf
