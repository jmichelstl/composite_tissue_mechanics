CPP=g++ --std=c++17 -O3 -Werror

ONEAPI_INC=-I/opt/intel/oneapi/2025.3/include/

ONEAPI_LINK=-L/opt/intel/oneapi/2025.3/lib/

MKL_LINK=-m64  -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lpthread -lm -ldl

VORO_INC=-I/usr/local/include/voro++

VTK_INC=-I/usr/local/include/vtk-9.5/

VTK_LINK=-L/usr/local/lib64 -Wl,--copy-dt-needed-entries \
    -lvtkCommonCore-9.5 -lvtkCommonDataModel-9.5 -lvtkIOXML-9.5

all:
	make nmcbatch
	make ngrelax
	make nmtraj_viz
	make nmc_stats
	make eq

lib/libflists.a:
	mkdir -p lib
	$(CPP) -c src/file_lists.cpp -o lib/libflists.a

lib/libgeomprim.a:
	mkdir -p lib
	$(CPP) -c src/geometric_primitives.cpp -o lib/libgeomprim.a

lib/librandompointproc.a:
	mkdir -p lib
	$(CPP) $(ONEAPI_INC) $(VORO_INC) -c src/random_point_processes.cpp -o lib/librandompointproc.a 

lib/libnmc.a:
	mkdir -p lib
	$(CPP) $(VORO_INC) $(ONEAPI_INC) -c src/network_mesh_complex.cpp -o lib/libnmc.a

lib/libnmio.a:
	mkdir -p lib
	$(CPP) -c -o lib/libnmio.a src/network_mesh_io.cpp

lib/libngru.a:
	mkdir -p lib
	$(CPP) $(ONEAPI_INC) -c -o lib/libngru.a src/network_gel_relax_utils.cpp

lib/libnmbvh.a:
	mkdir -p lib
	$(CPP) $(ONEAPI_INC) -c -o lib/libnmbvh.a src/nmbvh.cpp $(ONE_API_LINK) -ltbb

lib/libpredicates.a:
	mkdir -p lib
	gcc -O3 -Werror -c src/predicates.c -o lib/libpredicates.a

nmcbatch: lib/libgeomprim.a lib/libnmc.a lib/libnmio.a lib/librandompointproc.a lib/libnmbvh.a lib/libpredicates.a
	$(CPP) $(ONEAPI_INC) -o nmcbatch src/network_mesh_batch.cpp -ltinyxml2 -L/usr/lib/x86_64-linux-gnu/ -lgmsh -L./lib -lgeomprim -lnmc -lnmio -lnmbvh -lz -lvoro++ -lrandompointproc $(ONEAPI_LINK) -ltbb -lpredicates -lmetis

ngrelax: lib/libgeomprim.a lib/libflists.a lib/libnmio.a lib/libnmc.a lib/librandompointproc.a lib/libngru.a lib/libnmbvh.a lib/libpredicates.a
	$(CPP) $(ONEAPI_INC) -o ngrelax src/relax_network_gel_composite.cpp -L./lib -lgeomprim -lnmc -lnmio -lflists -lrandompointproc -lngru -lnmbvh -lgmsh -lz $(ONEAPI_LINK) -ltbb $(MKL_LINK) -ltinyxml2 -lboost_filesystem -lboost_regex -lvoro++ -lcholmod -lpredicates -lmatheval -lmetis

nmtraj_viz: lib/libgeomprim.a lib/libnmio.a lib/libnmc.a lib/librandompointproc.a lib/libnmbvh.a lib/libpredicates.a
	$(CPP) $(VTK_INC) -o nmtraj_viz src/visualize_network_mesh.cpp -L./lib -lgeomprim -lnmc -lnmio -lrandompointproc -lnmbvh -lgmsh -lz -lvoro++ $(VTK_LINK) $(ONEAPI_LINK) -ltbb -lpredicates -lmetis

nmc_stats: lib/libgeomprim.a lib/libnmio.a lib/libnmbvh.a lib/libpredicates.a
	$(CPP) -o nmc_stats src/compute_network_mesh_statistics.cpp -L./lib -lgeomprim -lnmio -lnmbvh -lz -ltinyxml2 $(ONEAPI_LINK) -ltbb -lpredicates -lmetis

eq: lib/libngru.a
	$(CPP) $(ONEAPI_INC) -o eq src/evaluate_equilibrium_swelling.cpp -L./lib -lngru -lgeomprim -lmatheval $(ONEAPI_LINK) -ltbb $(MKL_LINK) -lcholmod -lnmio -lz -ltinyxml2

geomclean:
	rm lib/libgeomprim.a

flistclean:
	rm lib/libflists.a

nmcclean:
	rm lib/libnmc.a

ioclean:
	rm -f lib/libnmio.a

rppclean:
	rm -f lib/librandompointproc.a

batchclean:
	rm -f nmcbatch

ngclean:
	rm -f ngrelax

vizclean:
	rm -f nmtraj_viz

statclean:
	rm -f nmc_stats

bvhclean:
	rm -f lib/libnmbvh.a

ngruclean:
	rm -f lib/libngru.a

eqclean:
	rm -f eq

clean:
	rm -f lib/*
	rm -f nmcbatch
	rm -f nglreax
	rm -f nmtraj_viz
	rm -f nmc_stats
	rm -f eq
