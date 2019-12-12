#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cmath>
#include <vector> 
#include <random>

struct system_model {
    int num_species;
    int num_reactions;
    int num_elements;
    double *Pre;
    double *Post;
    double *S;
    double *state;
    double *rates;
    double *tspan;
    int seed;
};

void PyArray_FromVector(PyObject *target, const std::vector<double> &source) {
    /*
    Copy elements rom a vector to a PyArray
    - Pyobject should be a PyArray owning its memory
    - memory should be of the same length as source
    */
    double *target_ptr = (double*)PyArray_DATA(target);
    for (int i = 0; i < source.size(); i++) {
        target_ptr[i] = source[i];
    }
    return;
}

static PyObject *PyArray_NewFromVector(const std::vector<double> &source, std::vector<npy_intp> &dims);
static PyObject *PyArray_NewFromVector(const std::vector<double> &source);

void gillespie_fun (std::vector<double> &time, std::vector<double> &state_history, std::vector<double> &events, std::vector<double> &total_propensity, system_model &sys) ;
void update_propensity(std::vector<double> &propensity, std::vector<double> &state, system_model &sys);
inline double comb_factor (int n, int k);
bool next_reaction (double *delta_t, int *index, system_model &sys, std::vector<double> &propensity, double rand_1, double rand_2);
void update_state(std::vector<double> &state,int index,system_model &sys);
void update_history(std::vector<double> &state_history,std::vector<double> &state, system_model &sys);
void update_total_propensity(std::vector<double> &total_propensity, const std::vector<double> &propensity, system_model &sys, double delta_t );

extern "C"{

static PyObject *simulate(PyObject *self, PyObject *args) {

    // set up requried data objects
    PyObject *pre_in = NULL, *post_in = NULL, *rates_in = NULL, *initial_in = NULL, *tspan_in = NULL;
    int seed;

    if (!PyArg_ParseTuple(args, "OOOOOi", 
                                &pre_in,
                                &post_in,
                                &rates_in,
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

    // parse rates to array
    PyObject *rates = PyArray_FROM_OTF(rates_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (rates == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        return NULL;
    }

    // parse initial to array
    PyObject *initial = PyArray_FROM_OTF(initial_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (initial == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(rates);
        return NULL;
    }

    // parse tspan to array
    PyObject *tspan = PyArray_FROM_OTF(tspan_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (tspan == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(rates);
        Py_DECREF(initial);
        return NULL;
    }

    // create stochiometry matrix
    double *pre_ptr = (double*)PyArray_DATA(pre);
    double *post_ptr = (double*)PyArray_DATA(post);
    std::vector<double> stoch(num_elements);
    for (int i = 0; i < stoch.size(); i++) {
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
    sys.rates = (double*)PyArray_DATA(rates);
    sys.tspan = (double*)PyArray_DATA(tspan);
    sys.seed = seed;  

    // for (int i = 0; i < num_reactions; i++){
    //     for (int j = 0; j < num_species; j++) {
    //         int ind = i*num_species+j;
    //         std::cout << sys.S[ind] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // construct intermediate storage
    size_t size_estimate = 1000;
    std::vector<double> time;
    time.reserve(size_estimate);
    std::vector<double> state_history;
    state_history.reserve(size_estimate*sys.num_species);
    std::vector<double> events(sys.num_reactions);
    std::vector<double> total_propensity(sys.num_reactions);
    
    // perform calculation
    gillespie_fun(time, state_history, events, total_propensity, sys); 
    
    // // create output matrix for time
    // plhs[0] = mxCreateDoubleMatrix(1,time.size(),mxREAL);
    // double *time_out = mxGetPr(plhs[0]);
    // for (int i = 0; i < time.size(); i++) {
    //     time_out[i] = time[i];
    // }
    
    // // create output matrix for the state
    // plhs[1] = mxCreateDoubleMatrix(num_species,state_history.size()/num_species,mxREAL);
    // double *state_out = mxGetPr(plhs[1]);
    // for (int i = 0; i < state_history.size(); i++) {
    //     state_out[i] = state_history[i];
    // }

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
    Py_DECREF(rates);
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

static PyMethodDef gillespie_methods[] = {
    //...
    {"simulate",  simulate, METH_VARARGS,
     "Stochastic simulation of a mass action model for a fixed initial over a given time span"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef gillespie_module = {
    PyModuleDef_HEAD_INIT,
    "tasep",   /* name of module */
    "Product Bernoulli Tasep ODE function", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module
                 or -1 if the module keeps state in global variables. */
    gillespie_methods
};

PyMODINIT_FUNC PyInit_gillespie(void) {
    import_array();
    return PyModule_Create(&gillespie_module);
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

void gillespie_fun (std::vector<double> &time, std::vector<double> &state_history, std::vector<double> &events, std::vector<double> &total_propensity, system_model &sys) {
    // preparations
    double t = sys.tspan[0];
    double t_max = sys.tspan[1];
    double delta_t = 0.0;
    int index = 0;
    std::vector<double> state(sys.state,sys.state+sys.num_species);
    std::vector<double> propensity(sys.num_reactions);
    std::mt19937 rng(sys.seed);
    std::uniform_real_distribution<double> U;
    // store initals in history vector
    time.push_back(t);
    update_history(state_history,state,sys);
    // sample path
    while (t < t_max) {
        // determine next event
        update_propensity(propensity,state,sys);
        bool success = next_reaction(&delta_t,&index,sys,propensity,U(rng),U(rng));
        if (not success)
            break;
        // update system
        t += delta_t;
        update_state(state,index,sys);
        // update output statistics
        events[index] += 1.0;
        time.push_back(t);
        update_history(state_history, state, sys);
        update_total_propensity(total_propensity, propensity, sys, delta_t);
    }
    return;
}

void update_propensity(std::vector<double> &propensity, std::vector<double> &state, system_model &sys) {
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
    if (n < k)
        res = 0.0;
    else {
        res = 1.0;
        for (int i = 0; i < k; i++) {
            res *= n-i;
        }
    }
    return(res);
}

bool next_reaction (double *delta_t, int *index, system_model &sys, std::vector<double> &propensity, double rand_1, double rand_2) {
    /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
    // calculate the reaction hazards
    std::vector<double> hazard(sys.num_reactions);
    double total_hazard = 0.0;
    for (size_t i = 0; i < sys.num_reactions; i++ ) {
        total_hazard += sys.rates[i]*propensity[i];
        hazard[i] = total_hazard;
    }
    // calculate reaction time
    *delta_t = -std::log(rand_1)/total_hazard;
    // sample random event from the individual hazards
    rand_2 *= total_hazard;
    *index = 0;
    while ( hazard[*index] < rand_2) {
        (*index)++;
    }
    return( total_hazard>0.0);
}

void update_state(std::vector<double> &state,int index,system_model &sys) {
    for (size_t i = 0; i < sys.num_species; i++ ) {
        size_t ind = index*sys.num_species+i;
        state[i] += sys.S[ind];
    }
    return;
}

void update_history(std::vector<double> &state_history,std::vector<double> &state, system_model &sys) {
    for (size_t i = 0; i < sys.num_species; i++) {
        state_history.push_back(state[i]);
    }
    return;
}

void update_total_propensity(std::vector<double> &total_propensity, const std::vector<double> &propensity, system_model &sys, double delta_t ) {
    for (size_t i = 0; i < sys.num_reactions; i++) {
        total_propensity[i] += delta_t*propensity[i];
    }
    return;
}



