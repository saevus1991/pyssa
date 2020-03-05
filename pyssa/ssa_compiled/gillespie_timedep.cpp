#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cmath>
#include <vector> 
#include <random>

struct system_model {
    unsigned num_species;
    unsigned num_reactions;
    unsigned num_elements;
    unsigned num_steps;
    double *Pre;
    double *Post;
    double *S;
    double *state;
    double *time_grid;
    double *control;
    double *tspan;
    double *rates;
    size_t seed;
};

void PyArray_FromVector(PyObject *target, const std::vector<double> &source) {
    /*
    Copy elements rom a vector to a PyArray
    - Pyobject should be a PyArray owning its memory
    - memory should be of the same length as source
    */
    double *target_ptr = (double*)PyArray_DATA(target);
    for (unsigned i = 0; i < source.size(); i++) {
        target_ptr[i] = source[i];
    }
    return;
}

static PyObject *PyArray_NewFromVector(const std::vector<double> &source, std::vector<npy_intp> &dims);
static PyObject *PyArray_NewFromVector(const std::vector<double> &source);

void gillespie_fun(system_model &sys, std::vector<double> &time, std::vector<double> &events, std::vector<double> &stats);
void update_propensity(std::vector<double> &propensity, std::vector<double> &state, const system_model &sys);
inline double comb_factor (int n, int k);
bool next_reaction(double *t, int *t_index, int *index, const system_model &sys, const std::vector<double> &propensity, const std::vector<double> &internal_time, std::vector<double> &stats);
void update_state(std::vector<double> &state,int index,const system_model &sys);
void construct_trajectory(const system_model &sys,const std::vector<double> events, std::vector<double> &trajectory);
void update_total_propensity(std::vector<double> &total_propensity, const std::vector<double> &propensity, system_model &sys, double delta_t );

extern "C"{

static PyObject *simulate(PyObject *self, PyObject *args) {

    // set up requried data objects
    PyObject *pre_in = NULL, *post_in = NULL, *control_in = NULL, *time_grid_in = NULL, *initial_in = NULL, *tspan_in = NULL;
    int seed;

    if (!PyArg_ParseTuple(args, "OOOOOOi", 
                                &pre_in,
                                &post_in,
                                &control_in,
                                &time_grid_in,
                                &initial_in,
                                &tspan_in,
                                &seed)) {
         return NULL;
    } 

    // parse pre matrix to array
    PyObject *pre = PyArray_FROM_OTF(pre_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (pre == NULL) {
        return NULL;
    }
    int pre_ndim = PyArray_NDIM(pre);
    if (pre_ndim != 2){
        Py_DECREF(pre);
        return NULL;
    }
    npy_intp* pre_dims = PyArray_DIMS(pre);

    // extract size informatoin
    int num_reactions = pre_dims[0];
    int num_species = pre_dims[1];
    int num_elements = PyArray_SIZE(pre);

    // parse post to array
    PyObject *post = PyArray_FROM_OTF(post_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (post == NULL) {
        Py_DECREF(pre);
        return NULL;
    }
    int post_ndim = PyArray_NDIM(post);
    if (post_ndim != 2){
        Py_DECREF(pre);
        Py_DECREF(post);
        return NULL;
    }

    // parse control to array
    PyObject *control = PyArray_FROM_OTF(control_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (control == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        return NULL;
    }

    // parse time_grid to array
    PyObject *time_grid = PyArray_FROM_OTF(time_grid_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (time_grid == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(control);
        return NULL;
    }
    npy_intp* time_grid_dims = PyArray_DIMS(time_grid);

    // get number of time steps
    int num_steps = time_grid_dims[0];

    // parse initial to array
    PyObject *initial = PyArray_FROM_OTF(initial_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (initial == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(control);
        Py_DECREF(time_grid);
        return NULL;
    }

    // parse tspan to array
    PyObject *tspan = PyArray_FROM_OTF(tspan_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (tspan == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(control);
        Py_DECREF(time_grid);
        Py_DECREF(initial);
        return NULL;
    }

    // create stochiometry matrix
    double *pre_ptr = (double*)PyArray_DATA(pre);
    double *post_ptr = (double*)PyArray_DATA(post);
    std::vector<double> stoch(num_elements);
    for (unsigned i = 0; i < stoch.size(); i++) {
        stoch[i] = post_ptr[i]-pre_ptr[i];
    }

    // create model structure
    system_model sys;
    sys.num_species = num_species;
    sys.num_reactions = num_reactions;
    sys.num_elements = num_elements;
    sys.Pre = pre_ptr;
    sys.Post = post_ptr;
    sys.S = stoch.data();
    sys.state = (double*)PyArray_DATA(initial);
    sys.control = (double*)PyArray_DATA(control);
    sys.time_grid = (double*)PyArray_DATA(time_grid);
    sys.num_steps = num_steps;
    sys.tspan = (double*)PyArray_DATA(tspan);
    sys.seed = seed;  


    // construct intermediate storage
    size_t size_estimate = 1000;
    std::vector<double> time;
    time.reserve(size_estimate);
    std::vector<double> events;
    events.reserve(size_estimate);
    std::vector<double> total_propensity(sys.num_reactions);
    std::vector<double> stats(sys.num_reactions);
    
    // perform calculation
    gillespie_fun(sys, time, events, stats);
    
    // // create output matrix for time
    // plhs[0] = mxCreateDoubleMatrix(1,time.size(),mxREAL);
    // double *time_out = mxGetPr(plhs[0]);
    // for (int i = 0; i < time.size(); i++) {
    //     time_out[i] = time[i];
    // }
    
    // create output matrix for the state
    std::vector<double> state_history(sys.num_species*events.size());
    construct_trajectory(sys, events, state_history);

    // create a copy of initial and tspan
    PyObject *initial_out = PyArray_NewCopy((PyArrayObject*)initial, NPY_CORDER);
    PyObject *tspan_out = PyArray_NewCopy((PyArrayObject*)tspan, NPY_CORDER);

    // create time output
    PyObject *time_out = PyArray_NewFromVector(time);

    // create state output
    PyObject *events_out = PyArray_NewFromVector(events);

    // create states output
    std::vector<npy_intp> state_history_dims(2);
    state_history_dims[0] = state_history.size()/num_species;
    state_history_dims[1] = num_species;  
    PyObject *state_history_out = PyArray_NewFromVector(state_history, state_history_dims);

    // construct output dictionary
    PyObject *sample = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O}",
                     "initial", initial_out,
                     "tspan", tspan_out,
                     "times", time_out,
                     "events", events_out,
                     "states", state_history_out);

    // clean up
    Py_DECREF(pre);
    Py_DECREF(post);
    Py_DECREF(control);
    Py_DECREF(time_grid);
    Py_DECREF(initial);
    Py_DECREF(tspan);
    Py_DECREF(initial_out);
    Py_DECREF(tspan_out);
    Py_DECREF(time_out);
    Py_DECREF(events_out);
    Py_DECREF(state_history_out);
    
    // return output dictionary
    return(sample);
}

static PyMethodDef gillespie_timedep_methods[] = {
    //...
    {"simulate",  simulate, METH_VARARGS,
     "Stochastic simulation of a mass action model for a fixed initial over a given time span"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef gillespie_timedep_module = {
    PyModuleDef_HEAD_INIT,
    "tasep",   /* name of module */
    "Product Bernoulli Tasep ODE function", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module
                 or -1 if the module keeps state in global variables. */
    gillespie_timedep_methods
};

PyMODINIT_FUNC PyInit_gillespie_timedep(void) {
    import_array();
    return PyModule_Create(&gillespie_timedep_module);
}

} // end extern C


static PyObject *PyArray_NewFromVector(const std::vector<double> &source, std::vector<npy_intp> &dims) {
    /*
    Create a new PyArray from an std::vector and store into empty PyObject
    - Pyobject should be NULL
    - output is constructed as 1d vector
    */
    // get dimension information
    PyObject *target = PyArray_SimpleNew(dims.size(), dims.data(), NPY_DOUBLE);
    PyArray_FromVector(target, source);
    return(target);
}

static PyObject *PyArray_NewFromVector(const std::vector<double> &source) {
    /*
    Create a new PyArray from an std::vector and store into empty PyObject
    - Pyobject should be NULL
    - output is constructed as 1d vector
    */
    // get dimension information
    std::vector<npy_intp> dims(1);
    dims[0] = source.size();   
    PyObject *target = PyArray_NewFromVector(source, dims);
    return(target);
}

void gillespie_fun(system_model &sys, std::vector<double> &time,std::vector<double> &events, std::vector<double> &stats) {
    // preparations
    double t = sys.tspan[0];
    double t_max = sys.tspan[1];
    int t_index = 0;
    int index = 0;
    std::vector<double> state(sys.state,sys.state+sys.num_species);
    std::vector<double> propensity(sys.num_reactions);
    std::mt19937 rng(sys.seed);
    std::uniform_real_distribution<double> U;
    // set up vectors for internal time
    std::vector<double> internal_time(sys.num_reactions);
    // initialize the internal times
    for (unsigned i = 0; i < sys.num_reactions; i++) {
        internal_time[i] = -std::log(U(rng));
    }
    // generate the sample path
    while (t < t_max) {
        //std::cout << t << std::endl;
        // update propensity
        update_propensity(propensity,state,sys);
        // compute next reaction event
        bool success = next_reaction(&t,&t_index,&index,sys,propensity,internal_time,stats);
        if (not success)
            break;
        // update system
        update_state(state,index,sys);
        // update internal time for the reaction that fired
        internal_time[index] += -std::log(U(rng));
        // save path variables
        time.push_back(t);
        events.push_back(index);
        //std::cout << t << " " << t_index << " " << internal_time[1] << " " << stats[1] << std::endl;
    }
}

// void update_propensity(std::vector<double> &propensity, std::vector<double> &state, const system_model &sys) {
//     // initialise to one
//     for (unsigned i = 0; i < sys.num_reactions; i++) {
//         propensity[i] = 1.0;
//     }
//     // calculate the stoichiometric factors
//     for (unsigned i = 0; i < sys.num_elements; i++) {
//         if ( sys.Pre[i] > 0 ) {
//             size_t species = i/sys.num_reactions;
//             size_t reaction = i%sys.num_reactions;
//             propensity[reaction] *= comb_factor(state[species],sys.Pre[i]);
//         }
//     }
//     return;
// }

void update_propensity(std::vector<double> &propensity, std::vector<double> &state, const system_model &sys) {
    // initialise to one
    for (int i = 0; i < sys.num_reactions; i++) {
        propensity[i] = 1.0;
    }
    // calculate the stoichiometric factors
    for (int i = 0; i < sys.num_reactions; i++) {
        for (int j = 0; j < sys.num_species; j++) {
            int ind = i*sys.num_species+j;
            propensity[i] *= comb_factor(state[j], sys.Pre[ind]);
        }
    }
    return;
}

inline double comb_factor (int n, int k) {
    double res;
    if (n <k) {
        res = 0.0;
    }
    else {
        res = 1.0;
        for (int i = 0; i < k; i++) {
            res *= n-i;
        }
    }
    return(res);
}

bool next_reaction (double *t, int *t_index, int *index, const system_model &sys, const std::vector<double> &propensity, const std::vector<double> &internal_time, std::vector<double> &stats) {
    /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
    *index = -1;
    // find the time_index that corresponds to largest time grid value smaller than the current time
    //    while ( *t_index < sys.time_grid.size() && sys.time_grid[*t_index] <= t )
    //        (*t_index)++;
    //    (*t_index)--;
    // initilize some variables for storing temporary information
    std::vector<double> tmp1(sys.num_reactions,0.0);
    std::vector<double> tmp2(sys.num_reactions,0.0);
    std::vector<double> tmp3(sys.num_reactions,0.0);
    // initialize tmp2 and tmp3 by the integral from the current time to the next larger grid step
    for (unsigned i = 0; i < sys.num_reactions; i++) {
        size_t ind = (*t_index)*sys.num_reactions;
        if (propensity[i] > 0) {
            tmp1[i] = (internal_time[i]-stats[i])/propensity[i];
            tmp2[i] = sys.control[ind+i]*(sys.time_grid[*t_index+1]-*t);
            tmp3[i] = tmp2[i];
        }
    }
    // for (int i = 0; i < sys.num_reactions; i++){
    //     std::cout << sys.control[i] << " " << tmp1[i] << " " << tmp2[i] << " " << tmp3[i] << std::endl;
    // }
    (*t_index)++;
    bool fired = false;
    // check if a reaction has already fired in the initial interval
    for (unsigned i = 0; i < sys.num_reactions; i++) {
        if ( propensity[i] > 0 && tmp2[i] > tmp1[i]) {
            fired = true;
            break;
        }
    }
    // iterate over all reactions and increase index until first reaction fires
    while ( *t_index < sys.num_steps-1 && !fired ) {
        // calculate the increment to the integral
        for (unsigned i = 0; i < sys.num_reactions; i++) {
            // if propensity is positive, evaluate the next time
            size_t ind = (*t_index)*sys.num_reactions;
            if (propensity[i] > 0) {
                tmp3[i] = sys.control[ind+i]*(sys.time_grid[*t_index+1]-sys.time_grid[*t_index]);
                tmp2[i] += tmp3[i];
            }
        }
        (*t_index)++;
        // check if any reaction has fired
        for (unsigned i = 0; i < sys.num_reactions; i++) {
            if ( propensity[i] > 0 && tmp2[i] > tmp1[i]) {
                fired = true;
                break;
            }
        }
    }
    // if a reaction has fired, undo the last update
    if ( fired ) {
        for (unsigned i = 0; i < sys.num_reactions; i++ ) {
            if (propensity[i] > 0) {
                tmp2[i] -= tmp3[i];
            }
        }
        (*t_index)--;
        // update the remainder term for the integral
        for (unsigned i = 0; i < sys.num_reactions; i++) {
            if (propensity[i] > 0) {
                tmp1[i] -= tmp2[i];
            }
        }
        // the remaining integral is linear in time and can be inverted analytically
        double min_time = sys.tspan[1];
        size_t ind = (*t_index)*sys.num_reactions;
        for (unsigned i = 0; i < sys.num_reactions; i++) {
            if (propensity[i] > 0 && tmp1[i] < tmp3[i]) { // the second condition ensures that a positive solution exists
                // solve for time
                double tmp_time = tmp1[i]/sys.control[ind+i];
                // store reaction index
                if ( tmp_time < min_time) {
                    min_time = tmp_time;
                    *index = i;
                }
            }
        }
        // calculate the time of the next reaction
        double delta_t = std::max(0.0,sys.time_grid[*t_index]-*t);
        *t += delta_t+min_time;
        // update the internal time for all the reactions
        for (unsigned i = 0; i < sys.num_reactions; i++ ) {
            stats[i] += (tmp2[i]+sys.control[ind+i])*min_time*propensity[i];
        }
    }
    else { // update the statistics up to the final time
        for ( unsigned i = 0; i < sys.num_reactions; i++) {
            stats[i] += tmp2[i]*propensity[i];
        }
    }
    return(fired);
}

void update_state(std::vector<double> &state, int index, const system_model &sys) {
    for (size_t i = 0; i < sys.num_species; i++ ) {
        size_t ind = index*sys.num_species+i;
        state[i] += sys.S[ind];
    }
    return;
}

// void update_history(std::vector<double> &state_history,std::vector<double> &state, system_model &sys) {
//     for (size_t i = 0; i < sys.num_species; i++) {
//         state_history.push_back(state[i]);
//     }
//     return;
// }

void construct_trajectory(const system_model &sys,const std::vector<double> events, std::vector<double> &trajectory) {
    // get initial state
    std::vector<double> state(sys.state,sys.state+sys.num_species);
    // fill up the trajectory by iterating over the events
    for ( int i = 0; i < events.size(); i++ ) {
        // update the state
        int index = events[i];
        update_state(state,index,sys);
        // append to the vector
        for (int j = 0; j < sys.num_species; j++) {
            trajectory[i*sys.num_species+j] = state[j];
        }
    }
}

// void update_total_propensity(std::vector<double> &total_propensity, const std::vector<double> &propensity, system_model &sys, double delta_t ) {
//     for (size_t i = 0; i < sys.num_reactions; i++) {
//         total_propensity[i] += delta_t*propensity[i];
//     }
//     return;
// }



