/*
 * This code provides functionality for reporting, importing, and visualizing
 * network-mesh composites used for simulating heterogeneous tissue mechanics.
 * There is support for both the initial state of network-mesh data structures
 * and recording and visualizing final states after simulations.
 */

#include <sstream>
#include "network_mesh_io.hpp"
#define CHUNK 16384

#define min(a,b) a < b ? a : b
#define max(a,b) a > b ? a : b

using namespace std;
using namespace tinyxml2;

string xml_header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n";

//Deflate data in an array, write it to a specified file, and indicate how
//much compressed data was written.
int def(void *src, size_t size, FILE *dest, unsigned *num_written){
    int ret, flush, counter;
    unsigned have, to_read;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK) return ret;

    *num_written = 0;
    to_read = size;
    counter = 0;

    /* compress until end of file */
    do {
        memcpy(in, (void *) ((char *) src + counter++ * CHUNK), min(CHUNK, to_read));
        strm.avail_in = min(CHUNK, to_read);
        to_read -= min(CHUNK, to_read);
        flush = !to_read ? Z_FINISH : Z_NO_FLUSH;
        strm.next_in = in;

        /* run deflate() on input until output buffer not full, finish
           compression if all of source has been read in */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = deflate(&strm, flush);    /* no bad return value */
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            have = CHUNK - strm.avail_out;
            *num_written += have;
            if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
                (void)deflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);
        assert(strm.avail_in == 0);     /* all input will be used */

        /* done when last data in file processed */
    } while (flush != Z_FINISH);
    assert(ret == Z_STREAM_END);        /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);
    return Z_OK;
}

//This is the similar to the above function, but here data are written to
//a buffer.
int def_to_buf(void *src, size_t size, void **dest, size_t buf_size, unsigned *num_written){
    int ret, flush, counter, capacity;
    unsigned have, to_read;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];
    void *new_buf;

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK) return ret;

    *num_written = 0;
    to_read = size;
    counter = 0;
    capacity = buf_size;

    /* compress until end of file */
    do {
        memcpy(in, (void *) ((char *) src + counter++ * CHUNK), min(CHUNK, to_read));
        strm.avail_in = min(CHUNK, to_read);
        to_read -= min(CHUNK, to_read);
        flush = !to_read ? Z_FINISH : Z_NO_FLUSH;
        strm.next_in = in;

        /* run deflate() on input until output buffer not full, finish
           compression if all of source has been read in */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = deflate(&strm, flush);    /* no bad return value */
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            have = CHUNK - strm.avail_out;
            capacity -= have;
            if(capacity < 0){
                while(capacity < 0){
		            capacity += buf_size;
		            buf_size *= 2;
                }
                new_buf = realloc(*dest, buf_size);
		        if(! new_buf){
                    cerr << "The buffer could not be expanded.\n";
                    (void)deflateEnd(&strm);
		            return Z_ERRNO; 
                }
		        *dest = new_buf;
            }
	        memcpy((void *) ((char *) *dest + *num_written), (const void *) out, have);
            *num_written += have;
        } while (strm.avail_out == 0);
        assert(strm.avail_in == 0);     /* all input will be used */

        /* done when last data in file processed */
    } while (flush != Z_FINISH);
    assert(ret == Z_STREAM_END);        /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);
    return Z_OK;
}

//Inflate compressed data
int inf(FILE *source, unsigned size, void *dest)
{
    int ret, counter;
    unsigned have, to_read;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);
    if (ret != Z_OK){
        return ret;
    }

    to_read = size;
    counter = 0;

    /* decompress until deflate stream ends or end of file */
    do {
        fread(in, sizeof(char), min(CHUNK, to_read), source);
        strm.avail_in = min(CHUNK, to_read);
        to_read -= min(CHUNK, to_read);
        strm.next_in = in;

        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR;     /* and fall through */
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                (void)inflateEnd(&strm);
                cerr << "Failed at second check.\n";
                return ret;
            }
            have = CHUNK - strm.avail_out;
            memcpy((void *) ((char *) dest + counter), out, have);
            counter += have;
        } while (strm.avail_out == 0);

        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);

    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

//Read the number of points, fibers, and tetrahedra in a network, and attempt
//to read data describing these entities. Return true upon success, and false
//upon failing to obtain all data.
bool extract_network_mesh_complex(char *file_name, vector<double> &coordinates, vector<int> &edges, vector<int> &triples, vector<double> &angles, vector<int> &tets){

    FILE *src;
    unsigned num_pts, pt_size, num_edges, edge_size, num_trips, trip_size;
    unsigned ang_size, num_tets, tet_size, num_read, size_buf[3];
    int inf_result;

    src = fopen(file_name, "r");
    if(src == NULL){
        cerr << "The NMC file could not be opened for reading.\n";
        return false;
    }

    //Attempt to read information about coordinates
    if((num_read = fread(size_buf, sizeof(unsigned), 2, src)) != 2){
        cerr << "Information about points could not be read.\n";
    }
    else{
        num_pts = size_buf[0];
        pt_size = size_buf[1];;
    }

    //Attempt to read information about edges
    if((num_read = fread(size_buf, sizeof(unsigned), 2, src)) != 2){
        cerr << "Information about edges could not be read.\n";
    }
    else{
        num_edges = size_buf[0];
        edge_size = size_buf[1];;
    }

    //Attempt to read information about triples
    if((num_read = fread(size_buf, sizeof(unsigned), 3, src)) != 3){
        cerr << "Information about triples could not be read.\n";
    }
    else{
        num_trips = size_buf[0];
        trip_size = size_buf[1];;
        ang_size = size_buf[2];;
    }

    //Attempt to read information about tetrahedra
    if((num_read = fread(size_buf, sizeof(unsigned), 2, src)) != 2){
        cerr << "Information about tetrahedra could not be read.\n";
    }
    else{
        num_tets = size_buf[0];
        tet_size = size_buf[1];;
    }

    //Attempt to extract coordinates
    coordinates = vector<double>(num_pts*3, 0);
    if((inf_result = inf(src, pt_size, &coordinates[0])) != Z_OK){
        cerr << "Extraction of the coordinates failed.\n";
        fclose(src);
        return false;
    }

    //Attempt to extract edges
    if(num_edges){
        edges = vector<int>(num_edges*2, 0);
        if((inf_result = inf(src, edge_size, &edges[0])) != Z_OK){
            cerr << "Extraction of the edges failed.\n";
            fclose(src);
            return false;
        }
    }

    //Attempt to extract triples and angles
    if(num_trips){
        triples = vector<int>(num_trips*3, 0);
        angles = vector<double>(num_trips, 0);
        if((inf_result = inf(src, trip_size, &triples[0])) != Z_OK){
            cerr << "Extraction of the triples failed.\n";
            fclose(src);
            return false;
        }
        if((inf_result = inf(src, ang_size, &angles[0])) != Z_OK){
            cerr << "Extraction of the angles failed.\n";
            fclose(src);
            return false;
        }
    }

    //Attempt to extract tetrahedra
    if(num_tets){
        tets = vector<int>(num_tets*4, 0);
        if((inf_result = inf(src, tet_size, &tets[0])) != Z_OK){
            cerr << "Extraction of the tetrahedra failed.\n";
            fclose(src);
            return false;
        }
    }

    fclose(src);
    return true;
}

bool report_network_mesh_complex(char *file_name, shared_ptr<NetworkMeshComplex> nmc){

    FILE *output;
    unsigned raw_size, to_write, comp_coord_size, comp_edge_size,comp_trip_size;
    unsigned comp_ang_size, comp_tet_size, goose_egg = 0;
    void *coord_buf, *edge_buf, *trip_buf, *ang_buf, *tet_buf;

    edge_buf = NULL;
    trip_buf = NULL;
    ang_buf = NULL;
    tet_buf = NULL;

    comp_coord_size = comp_edge_size = comp_trip_size = comp_ang_size = comp_tet_size = 0;

    output = fopen(file_name, "w");

    if(output == NULL){
        cerr << "The output file could not be opened.\n";
        return false;
    }

    //Report coordinates
    vector<double> coordinates = nmc->get_points();
    if(coordinates.size() < 3){
        cerr << "There must be a non-zero number of points.\n";
	    fclose(output);
	    return false;
    }

    raw_size = coordinates.size() / 3;
    fwrite(&raw_size, sizeof(unsigned), 1, output);

    to_write = 3*raw_size*sizeof(double);
    coord_buf = malloc(to_write);
    if(! coord_buf){
        cerr << "The coordinate buffer could not be allocated.\n";
	    fclose(output);
	    return false;
    }
    if(def_to_buf(&coordinates[0], to_write, &coord_buf, to_write, &comp_coord_size) != Z_OK){
        cerr << "Error writing coordinates.\n";
	    free(coord_buf);
	    fclose(output);
	    return false;
    }
    fwrite(&comp_coord_size, sizeof(unsigned), 1, output);

    //Report edges
    vector<edge> edges = nmc->get_edges();
    if(edges.size() > 0){
        vector<int> edges_simple;
        for(edge e :  edges){
            edges_simple.push_back(get<0>(e));
            edges_simple.push_back(get<1>(e));
        }

        raw_size = edges_simple.size() / 2;
        fwrite(&raw_size, sizeof(unsigned), 1, output);
        to_write = 2*raw_size*sizeof(int);
        edge_buf = malloc(to_write);
        if(! edge_buf){
            cerr << "The edge buffer could not be allocated.\n";
	        free(coord_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&edges_simple[0], to_write, &edge_buf, to_write, &comp_edge_size) != Z_OK){
            cerr << "Error writing edges.\n";
	        free(coord_buf);
            free(edge_buf);
            fclose(output);
            return false;
         }
         fwrite(&comp_edge_size, sizeof(unsigned), 1, output);
    }
    else{
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
    }

    //Report triples and angles
    vector<triple> trips = nmc->get_triples();
    if(trips.size() > 0){
        vector<int> trips_simple;
        for(triple t :  trips){
            trips_simple.push_back(get<0>(t));
            trips_simple.push_back(get<1>(t));
            trips_simple.push_back(get<2>(t));
        }
        raw_size = trips_simple.size() / 3;
        fwrite(&raw_size, sizeof(unsigned), 1, output);
        to_write = 3*raw_size*sizeof(int);
        trip_buf = malloc(to_write);
        if(! trip_buf){
            cerr << "The triple buffer could not be allocated.\n";
	        free(coord_buf);
            free(edge_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&trips_simple[0], to_write, &trip_buf, to_write, &comp_trip_size) != Z_OK){
            cerr << "Error writing edges.\n";
            free(coord_buf);
            free(edge_buf);
	        free(trip_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_trip_size, sizeof(unsigned), 1, output);

        vector<double> angles = nmc->get_angles();
        to_write = raw_size*sizeof(double);
        ang_buf = malloc(to_write);
        if(! ang_buf){
            cerr << "The angle buffer could not be allocated.\n";
	        free(coord_buf);
            free(edge_buf);
	        free(trip_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&angles[0], to_write, &ang_buf, to_write, &comp_ang_size) != Z_OK){
            cerr << "Error writing angles.\n";
            free(coord_buf);
            free(edge_buf);
	        free(trip_buf);
	        free(ang_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_ang_size, sizeof(unsigned), 1, output);
    }
    else{
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
    }

    //Report tetrahedra
    vector<int> tets = nmc->get_tetrahedra();
    if(tets.size() > 0){
        raw_size = tets.size() / 4;
        fwrite(&raw_size, sizeof(unsigned), 1, output);
        to_write = 4*raw_size*sizeof(int);
        tet_buf = malloc(to_write);
        if(! tet_buf){
            cerr << "The tetrahedron buffer could not be allocated.\n";
	        free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(trip_buf) free(ang_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&tets[0], to_write, &tet_buf, to_write, &comp_tet_size) != Z_OK){
            cerr << "Error writing tetrahedra.\n";
            free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(ang_buf) free(ang_buf);
	        free(tet_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_tet_size, sizeof(unsigned), 1, output);
    }
    else{
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
    }

    fwrite(coord_buf, sizeof(char), comp_coord_size, output);
    if(edge_buf){
        fwrite(edge_buf, sizeof(char), comp_edge_size, output);
    }
    if(trip_buf){
        fwrite(trip_buf, sizeof(char), comp_trip_size, output);
        fwrite(ang_buf, sizeof(char), comp_ang_size, output);
    }
    if(tet_buf){
        fwrite(tet_buf, sizeof(char), comp_tet_size, output);
    }

    free(coord_buf);
    if(edge_buf) free(edge_buf);
    if(trip_buf) free(trip_buf);
    if(ang_buf) free(ang_buf);
    if(tet_buf) free(tet_buf);
    fclose(output);

    return true;
}

//TODO: Finish modifying this function to account for differences in data structures
bool report_disjoint_network_mesh_complex(char *file_name, shared_ptr<RandomNMCBase> rnmc){

    FILE *output;
    unsigned raw_size, to_write, comp_coord_size, comp_edge_size,comp_trip_size;
    unsigned comp_ang_size, comp_tet_size, comp_vr_size, cv_size, goose_egg = 0;
    void *coord_buf, *edge_buf, *trip_buf, *ang_buf, *tet_buf;
    void *vert_range_buf, *contained_vert_buf;

    edge_buf = NULL;
    trip_buf = NULL;
    ang_buf = NULL;
    tet_buf = NULL;

    comp_coord_size = comp_edge_size = comp_trip_size = comp_ang_size = comp_tet_size = 0;

    output = fopen(file_name, "w");

    if(output == NULL){
        cerr << "The output file could not be opened.\n";
        return false;
    }

    //Report coordinates
    vector<double> coordinates = rnmc->get_points();
    if(coordinates.size() < 3){
        cerr << "There must be a non-zero number of points.\n";
	    fclose(output);
	    return false;
    }

    raw_size = coordinates.size() / 3;
    fwrite(&raw_size, sizeof(unsigned), 1, output);

    to_write = 3*raw_size*sizeof(double);
    coord_buf = malloc(to_write);

    //Note number of network vertices
    raw_size = rnmc->get_num_net_verts();
    fwrite(&raw_size, sizeof(unsigned), 1, output);

    if(! coord_buf){
        cerr << "The coordinate buffer could not be allocated.\n";
	    fclose(output);
	    return false;
    }
    if(def_to_buf(&coordinates[0], to_write, &coord_buf, to_write, &comp_coord_size) != Z_OK){
        cerr << "Error writing coordinates.\n";
	    free(coord_buf);
	    fclose(output);
	    return false;
    }
    fwrite(&comp_coord_size, sizeof(unsigned), 1, output);

    //Report edges
    vector<edge> edges = rnmc->get_edges();
    if(edges.size() > 0){
        vector<int> edges_simple;
        for(edge e :  edges){
            edges_simple.push_back(get<0>(e));
            edges_simple.push_back(get<1>(e));
        }

        raw_size = edges_simple.size() / 2;
        fwrite(&raw_size, sizeof(unsigned), 1, output);
        to_write = 2*raw_size*sizeof(int);
        edge_buf = malloc(to_write);
        if(! edge_buf){
            cerr << "The edge buffer could not be allocated.\n";
	        free(coord_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&edges_simple[0], to_write, &edge_buf, to_write, &comp_edge_size) != Z_OK){
            cerr << "Error writing edges.\n";
	        free(coord_buf);
            free(edge_buf);
            fclose(output);
            return false;
         }
         fwrite(&comp_edge_size, sizeof(unsigned), 1, output);
    }
    else{
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
    }

    //Report triples and angles
    vector<triple> trips = rnmc->get_triples();
    if(trips.size() > 0){
        vector<int> trips_simple;
        for(triple t :  trips){
            trips_simple.push_back(get<0>(t));
            trips_simple.push_back(get<1>(t));
            trips_simple.push_back(get<2>(t));
        }
        raw_size = trips_simple.size() / 3;
        fwrite(&raw_size, sizeof(unsigned), 1, output);
        to_write = 3*raw_size*sizeof(int);
        trip_buf = malloc(to_write);
        if(! trip_buf){
            cerr << "The triple buffer could not be allocated.\n";
	        free(coord_buf);
            free(edge_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&trips_simple[0], to_write, &trip_buf, to_write, &comp_trip_size) != Z_OK){
            cerr << "Error writing edges.\n";
            free(coord_buf);
            free(edge_buf);
	        free(trip_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_trip_size, sizeof(unsigned), 1, output);

        vector<double> angles = rnmc->get_angles();
        to_write = raw_size*sizeof(double);
        ang_buf = malloc(to_write);
        if(! ang_buf){
            cerr << "The angle buffer could not be allocated.\n";
	        free(coord_buf);
            free(edge_buf);
	        free(trip_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&angles[0], to_write, &ang_buf, to_write, &comp_ang_size) != Z_OK){
            cerr << "Error writing angles.\n";
            free(coord_buf);
            free(edge_buf);
	        free(trip_buf);
	        free(ang_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_ang_size, sizeof(unsigned), 1, output);
    }
    else{
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
    }

    //Report tetrahedra
    vector<int> tets = rnmc->get_tetrahedra();
    if(tets.size() > 0){
        raw_size = tets.size() / 4;
        fwrite(&raw_size, sizeof(unsigned), 1, output);
        to_write = 4*raw_size*sizeof(int);
        tet_buf = malloc(to_write);
        if(! tet_buf){
            cerr << "The tetrahedron buffer could not be allocated.\n";
	        free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(trip_buf) free(ang_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&tets[0], to_write, &tet_buf, to_write, &comp_tet_size) != Z_OK){
            cerr << "Error writing tetrahedra.\n";
            free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(ang_buf) free(ang_buf);
	        free(tet_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_tet_size, sizeof(unsigned), 1, output);

        //Report network vertices contained within tetrahedra
        vector<int> tet_ranges = rnmc->get_tet_ranges();
        to_write = tet_ranges.size()*sizeof(int);
        vert_range_buf = malloc(to_write);
        if(! vert_range_buf){
            cerr << "The tetrahedron buffer could not be allocated.\n";
	        free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(trip_buf) free(ang_buf);
            free(tet_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&tet_ranges[0], to_write, &vert_range_buf, to_write, &comp_vr_size) != Z_OK){
            cerr << "Error writing tetrahedra.\n";
            free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(ang_buf) free(ang_buf);
            free(tet_buf);
	        free(vert_range_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_vr_size, sizeof(unsigned), 1, output);

        vector<int> points_in_tets = rnmc->get_point_in_tet_records();
        to_write = points_in_tets.size()*sizeof(int);
        contained_vert_buf = malloc(to_write);
        if(! contained_vert_buf){
            cerr << "The tetrahedron buffer could not be allocated.\n";
	        free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(trip_buf) free(ang_buf);
            free(tet_buf);
            free(vert_range_buf);
            fclose(output);
            return false;
        }
        if(def_to_buf(&points_in_tets[0], to_write, &contained_vert_buf, to_write, &comp_cv_size) != Z_OK){
            cerr << "Error writing tetrahedra.\n";
            free(coord_buf);
            if(edge_buf) free(edge_buf);
	        if(trip_buf) free(trip_buf);
	        if(ang_buf) free(ang_buf);
	        free(tet_buf);
            free(vert_range_buf);
            free(contained_vert_buf);
            fclose(output);
	        return false;
        }
        fwrite(&comp_cv_size, sizeof(unsigned), 1, output);
    }
    else{
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
         fwrite(&goose_egg, sizeof(unsigned), 1, output);
    }

    fwrite(coord_buf, sizeof(char), comp_coord_size, output);
    if(edge_buf){
        fwrite(edge_buf, sizeof(char), comp_edge_size, output);
    }
    if(trip_buf){
        fwrite(trip_buf, sizeof(char), comp_trip_size, output);
        fwrite(ang_buf, sizeof(char), comp_ang_size, output);
    }
    if(tet_buf){
        fwrite(tet_buf, sizeof(char), comp_tet_size, output);
        fwrite(vert_range_buf, sizeof(char), comp_vr_size, output);
        fwrite(contained_vert_buf, sizeof(char), comp_cv_size, output);
    }

    free(coord_buf);
    if(edge_buf) free(edge_buf);
    if(trip_buf) free(trip_buf);
    if(ang_buf) free(ang_buf);
    if(tet_buf){
        free(tet_buf);
        free(vert_range_buf);
        free(contained_vert_buf);
    }
    fclose(output);

    return true;
}

bool extract_disjoint_network_mesh_complex(char *file_name, vector<double> &coordinates, unsigned &num_net_verts, vector<int> &edges, vector<int> &triples, vector<double> &angles, vector<int> &tets, vector<int> &vert_ranges, vector<int> &contained_verts){

    FILE *src;
    unsigned num_pts, pt_size, num_edges, edge_size, num_trips;
    unsigned trip_size, ang_size, num_tets, tet_size, vr_size, cv_size;
    unsigned num_read, size_buf[3];
    int inf_result;

    src = fopen(file_name, "r");
    if(src == NULL){
        cerr << "The NMC file could not be opened for reading.\n";
        return false;
    }

    //Attempt to read information about coordinates
    if((num_read = fread(size_buf, sizeof(unsigned), 3, src)) != 3){
        cerr << "Information about points could not be read.\n";
    }
    else{
        num_pts = size_buf[0];
        num_net_verts = size_buf[1];
        pt_size = size_buf[2];
    }

    //Attempt to read information about edges
    if((num_read = fread(size_buf, sizeof(unsigned), 2, src)) != 2){
        cerr << "Information about edges could not be read.\n";
    }
    else{
        num_edges = size_buf[0];
        edge_size = size_buf[1];;
    }

    //Attempt to read information about triples
    if((num_read = fread(size_buf, sizeof(unsigned), 3, src)) != 3){
        cerr << "Information about triples could not be read.\n";
    }
    else{
        num_trips = size_buf[0];
        trip_size = size_buf[1];;
        ang_size = size_buf[2];;
    }

    //Attempt to read information about tetrahedra
    if((num_read = fread(size_buf, sizeof(unsigned), 2, src)) != 2){
        cerr << "Information about tetrahedra could not be read.\n";
    }
    else{
        num_tets = size_buf[0];
        tet_size = size_buf[1];;
    }

    //If there are tetrahedra, read information about network vertices contained
    //in tetrahedra.
    if(num_tets > 0){
        if((num_read = fread(size_buf, sizeof(unsigned), 2, src)) != 2){
            cerr << "Information about tetrahedron vertex ranges and vertex membership could not be read.\n";
        }
        else{
            vr_size = size_buf[0];
            vc_size = size_buf[1];
        }
    }

    //Attempt to extract coordinates
    coordinates = vector<double>(num_pts*3, 0);
    if((inf_result = inf(src, pt_size, &coordinates[0])) != Z_OK){
        cerr << "Extraction of the coordinates failed.\n";
        fclose(src);
        return false;
    }

    //Attempt to extract edges
    if(num_edges){
        edges = vector<int>(num_edges*2, 0);
        if((inf_result = inf(src, edge_size, &edges[0])) != Z_OK){
            cerr << "Extraction of the edges failed.\n";
            fclose(src);
            return false;
        }
    }

    //Attempt to extract triples and angles
    if(num_trips){
        triples = vector<int>(num_trips*3, 0);
        angles = vector<double>(num_trips, 0);
        if((inf_result = inf(src, trip_size, &triples[0])) != Z_OK){
            cerr << "Extraction of the triples failed.\n";
            fclose(src);
            return false;
        }
        if((inf_result = inf(src, ang_size, &angles[0])) != Z_OK){
            cerr << "Extraction of the angles failed.\n";
            fclose(src);
            return false;
        }
    }

    //Attempt to extract tetrahedra
    if(num_tets){
        tets = vector<int>(num_tets*4, 0);
        if((inf_result = inf(src, tet_size, &tets[0])) != Z_OK){
            cerr << "Extraction of the tetrahedra failed.\n";
            fclose(src);
            return false;
        }

        vert_ranges = vector<int>(num_tets+1, 0);
        if((inf_result = inf(src, vr_size, &vert_ranges[0])) != Z_OK){
            cerr << "Extraction of the tetrahedron vertex ranges failed.\n";
            fclose(src);
            return false;
        }

        contained_verts = vector<int>(num_net_verts, 0);
        if((inf_result = inf(src, cv_size, &contained_verts[0])) != Z_OK){
            cerr << "Extraction of the tetrahedron vertex members failed.\n";
            fclose(src);
            return false;
        }
    }

    fclose(src);
    return true;
}

//Open a manifest describing a relaxation run
bool open_network_mesh_trajectory(string output, string nmc_file, RelaxParams params, string &xml_name, string &nmt_name){

    FILE *xml_handle;
    ostringstream name;

    name << output << ".xml";
    xml_name = name.str();

    xml_handle = fopen(xml_name.c_str(), "w");
    if(! xml_handle){
        cerr << "The XML manifest could not be opened.\n";
	    return false;
    }

    name.str("");
    name << output << ".nmt";
    nmt_name = name.str();

    //Open the document and write the parameters
    fprintf(xml_handle, xml_header.c_str());
    fprintf(xml_handle, "<NetworkMeshTrajectory>\n");

    fprintf(xml_handle, "\t<parameters>\n");
    fprintf(xml_handle, "\t\t<param name=\"stretch\" value=\"%1.10le\"/>\n", params.stretch);
    fprintf(xml_handle, "\t\t<param name=\"bend\" value=\"%1.10le\"/>\n", params.bend);
    fprintf(xml_handle, "\t\t<param name=\"gel_mod\" value=\"%1.10le\"/>\n", params.gel_mod);
    fprintf(xml_handle, "\t\t<param name=\"excess_osm\" value=\"%1.10le\"/>\n", params.excess_osm);
    fprintf(xml_handle, "\t\t<param name=\"u_thermal\" value=\"%1.10le\"/>\n", params.u_thermal);
    fprintf(xml_handle, "\t\t<param name=\"chi\" value=\"%1.10le\"/>\n", params.chi);
    fprintf(xml_handle, "\t</parameters>\n");

    //Open the series of relaxation steps
    fprintf(xml_handle, "\n<relaxation_run reference_file=\"%s\">\n", nmc_file.c_str());
    fclose(xml_handle);

    return true;
}

//Append a state of the network-mesh complex
bool append_network_mesh_state(Vec3D disp, vector<double> coordinates, string xml_name, string nmt_name){

    unsigned num_written;
    FILE *xml_handle, *nmt_handle;

    xml_handle = fopen(xml_name.c_str(), "a");
    if(! xml_handle){
        cerr << "The XML manifest could not be opened for appending.\n";
	    return false;
    }

    nmt_handle = fopen(nmt_name.c_str(), "a");
    if(! nmt_handle){
        cerr << "The NMT file could not be opened for appending.\n";
	    fclose(xml_handle);
	    return false;
    }

    fprintf(xml_handle, "\t\t<step dx=\"%1.10le\" dy=\"%1.10le\" dz = \"%1.10le\" ", disp.x, disp.y, disp.z);

    if( def((void *) &coordinates[0], coordinates.size()*sizeof(double), nmt_handle, &num_written) != Z_OK){
        cerr << "Error writing coordinate data.\n";
	    return false;
    }
    fprintf(xml_handle, "xsize=\"%u\"/>\n", num_written); 

    /*if( def((void *) &energies[0], energies.size()*sizeof(double), nmt_handle, &num_written) != Z_OK){
        cerr << "Error writing energy data.\n";
	return false;
    }
    fprintf(xml_handle, "esize=\"%d\"/>\n", num_written);*/

    fclose(xml_handle);
    fclose(nmt_handle);

    return true;
}

//Close a manifest describing a relaxation run
bool close_network_mesh_trajectory(string xml_name){

    FILE *xml_handle = fopen(xml_name.c_str(), "a");

    if(! xml_handle){
        cerr << "The XML file could not be opened to close the trajectory record.\n";
	    return false;
    }

    fprintf(xml_handle, "</relaxation_run>\n\n</NetworkMeshTrajectory>\n");
    fclose(xml_handle);

    return true;
}

//Parse a parameter entry
bool parse_nmc_param(XMLElement *param_elem, string &name, double &value){

    XMLError result;
    const char *cname;

    //Look for the parameter name
    result = param_elem->QueryStringAttribute("name", &cname);
    if(result != XML_SUCCESS){
        cerr << "Error: a parameter name could not be read.\n";
	    return false;
    }
    name = string(cname);

    //Look for the parameter value
    result = param_elem->QueryDoubleAttribute("value", &value);
    if(result != XML_SUCCESS){
        cerr << "Error: the value for parameter \"" << name << "\" could not be read.\n";
	    return false;
    }

    return true;
}

//Parse a parameter block of a network-mesh trajectory file
bool parse_nmc_param_block(XMLElement *param_block, NMCRelaxationRun run){

    int param_count = 0;
    string name;
    double value;
    XMLElement *child;

    child = param_block->FirstChildElement("param");
    while(child != NULL && param_count < EXPECTED_PARAM_COUNT){

        if(! parse_nmc_param(child, name, value)){
            cerr << "Parsing of the parameter block failed.\n";
	        return false;
        }

	    if(name == "stretch"){
            run.params.stretch = value;
            param_count ++;
        }
	    else if(name == "bend"){
            run.params.bend = value;
            param_count ++;
        }
	    else if(name == "gel_mod"){
            run.params.gel_mod = value;
            param_count ++;
        }
	    else if(name == "excess_osm"){
            run.params.excess_osm = value;
            param_count ++;
        }
	    else if(name == "u_thermal"){
            run.params.u_thermal = value;
            param_count ++;
        }
	    else if(name == "chi"){
            run.params.chi = value;
            param_count ++;
        }
	    else{
            cerr << "Unrecognized parameter: " << name << endl;
	        return false;
        }

	    param_block->DeleteChild(child);
	    child = param_block->FirstChildElement("param");
    }

    if(param_count < EXPECTED_PARAM_COUNT){
        cerr << "Too few physical parameters were read.\n";
    }

    return param_count == EXPECTED_PARAM_COUNT;
}

//Parse information about the saved state of a network
bool parse_relaxation_step(XMLElement *step_elem, NMCRelaxationRun &run){

    XMLError result;
    double value;
    unsigned size;
    RelaxationStep rstep;

    //Attempt to read Cartesian displacement coordinates
    result = step_elem->QueryDoubleAttribute("dx", &value);
    if(result != XML_SUCCESS){
        cerr << "The x displacement could not be read.\n";
	    return false;
    }
    rstep.dx = value;

    result = step_elem->QueryDoubleAttribute("dy", &value);
    if(result != XML_SUCCESS){
        cerr << "The y displacement could not be read.\n";
	    return false;
    }
    rstep.dy = value;

    result = step_elem->QueryDoubleAttribute("dz", &value);
    if(result != XML_SUCCESS){
        cerr << "The z displacement could not be read.\n";
	    return false;
    }
    rstep.dz = value;

    //Attempt to read compressed coordinate and energy sizes
    result = step_elem->QueryUnsignedAttribute("xsize", &size);
    if(result != XML_SUCCESS){
        cerr << "The compressed coordinate size could not be read.\n";
	    return false;
    }
    rstep.xsize = size;

    /*result = step_elem->QueryUnsignedAttribute("esize", &size);
    if(result != XML_SUCCESS){
        cerr << "The compressed energy size could not be read.\n";
	    return false;
    }
    rstep.esize = size;*/

    run.steps.push_back(rstep);

    return true;
}

//Parse a block of relaxation steps
bool parse_relaxation_block(XMLElement *relaxation_elem, NMCRelaxationRun &run){

    XMLError result;
    XMLElement *child;
    const char *name;

    result = relaxation_elem->QueryStringAttribute("reference_file", &name);
    if(result != XML_SUCCESS){
        cerr << "No reference file was specified.\n";
        return false;
    }
    run.nmc_name = string(name);

    child = relaxation_elem->FirstChildElement("step");
    while(child != NULL){
        if(! parse_relaxation_step(child, run)){
            cerr << "The relaxation run block could not be parsed.\n";
            return false;
        }

        relaxation_elem->DeleteChild(child);
        child = relaxation_elem->FirstChildElement("step");
    }

    if(run.steps.size() == 0){
        cerr << "No relaxation steps were read.\n";
    }

    return run.steps.size() > 0;
}

//Import data about a relaxation run. Obtain physical parameters, descriptions
//of each relaxation step, and a handle to the file containing compressed,
//binary records of final positions and forces.
bool import_network_mesh_trajectory(string base_name, NMCRelaxationRun &run, FILE **nmt_handle){

    XMLDocument doc;
    XMLElement *root, *child;
    XMLError result; 
    ostringstream oss;

    //Make sure a handle to the binary file can be opened
    oss << base_name << ".nmt";
    *nmt_handle = fopen(oss.str().c_str(), "r");
    if(! *nmt_handle){
        cerr << "The binary trajectory file could not be opened.\n";
	    return false;
    }

    oss.str("");
    oss << base_name << ".xml";
    if(doc.LoadFile(oss.str().c_str()) != XML_SUCCESS){
        cerr << "The XML manifest could not be loaded.\n";
        return false;
    }

    root = doc.RootElement();

    //Make sure the document tree can be parsed, and the opening node is correct
    if(! root){
        cerr << "The DOM was empty.\n";
        return false;
    }

    else if(strcmp(root->Name(), "NetworkMeshTrajectory") != 0){
        cerr << "The root node must be ACCTrajectory\n";
        return false;
    }

    //Attempt to load physical parameters
    child = root->FirstChildElement("parameters");
    if(! child){
        cerr << "No parameter block was found.\n";
	    return false;
    }
    if(! parse_nmc_param_block(child, run)){
        cerr << "Parsing of parameters failed.\n";
	    return false;
    }

    //Attempt to load a record of relaxation steps
    child = root->FirstChildElement("relaxation_run");
    if(! child){
        cerr << "No relaxation run block was found.\n";
	    return false;
    }
    if(! parse_relaxation_block(child, run)){
        cerr << "Parsing the relaxation run failed.\n";
	    return false;
    }

    return true;
}


//Find the number of decimal digits used to represent an integer for the
//purpose of determining zero padding.
int get_dec_digits(int value){
    
    int dec_digits = 0;

    if(! value) return 1;
    
    while(value){
        dec_digits ++;
        value /= 10;
    }   
    
    return dec_digits;
}
