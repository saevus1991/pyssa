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
    double *Pre;
    double *Post;
    double *S;
    double *state;
    double *rates;
    double *tspan;
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

void gillespie_fun (system_model &sys, std::vector<double> &time, std::vector<double> &events, std::vector<double> &stats, double *llh) ;
void update_propensity(std::vector<double> &propensity, std::vector<double> &state, const system_model &sys);
inline double comb_factor (int n, int k);
bool next_reaction (double *t, int *index, system_model &sys, std::vector<double> &propensity, double rand_1, double rand_2, std::vector<double> &stats, double *reaction_llh);
void update_state(std::vector<double> &state,int index,const system_model &sys);
void update_history(std::vector<double> &state_history,std::vector<double> &state, system_model &sys);
void update_total_propensity(std::vector<double> &total_propensity, const std::vector<double> &propensity, system_model &sys, double delta_t );
void construct_trajectory(const system_model &sys,const std::vector<double> events, std::vector<double> &trajectory);
double compute_likelihood(const system_model &sys, double *events, double *times, unsigned num_events);

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
    std::vector<double> events;
    events.reserve(size_estimate);
    std::vector<double> stats(sys.num_reactions);
    double llh = 0.0;
    
    // perform calculation
    gillespie_fun(sys, time, events, stats, &llh); 

    // create output matrix for the state
    std::vector<double> state_history(sys.num_species*events.size());
    construct_trajectory(sys, events, state_history);
    
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

    // compute log likelihood
    for (int i = 0; i < stats.size(); i++) {
        llh -= stats[i];
    }

    //llh = compute_likelihood(sys, events.data(), time.data(), events.size());

    // construct output dictionary
    PyObject *sample = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:d}",
                     "initial", initial_out,
                     "tspan", tspan_out,
                     "times", time_out,
                     "events", events_out,
                     "states", state_history_out,
                     "llh", llh);

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

static PyObject *llh(PyObject *self, PyObject *args) {

    // set up requried data objects
    PyObject *pre_in = NULL, *post_in = NULL, *rates_in = NULL, *initial_in = NULL, *tspan_in = NULL, *times_in = NULL, *events_in = NULL;

    if (!PyArg_ParseTuple(args, "OOOOOOO", 
                                &pre_in,
                                &post_in,
                                &rates_in,
                                &initial_in,
                                &tspan_in,
                                &times_in,
                                &events_in)) {
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

    // parse times to array
    PyObject *times = PyArray_FROM_OTF(times_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (times == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(rates);
        Py_DECREF(initial);
        Py_DECREF(tspan);
        return NULL;
    }

    // get number of time steps
    npy_intp* times_dims = PyArray_DIMS(times);
    unsigned num_events = times_dims[0];

    // parse events to array
    PyObject *events = PyArray_FROM_OTF(events_in, NPY_DOUBLE, NPY_IN_ARRAY);
    if (events == NULL) {
        Py_DECREF(pre);
        Py_DECREF(post);
        Py_DECREF(rates);
        Py_DECREF(initial);
        Py_DECREF(tspan);
        Py_DECREF(times);
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
    sys.rates = (double*)PyArray_DATA(rates);
    sys.tspan = (double*)PyArray_DATA(tspan);

    // compute log likelihood
    double llh = compute_likelihood(sys, (double*)PyArray_DATA(events), (double*)PyArray_DATA(times), num_events);

    // Build output value
    PyObject *llh_out = Py_BuildValue("d", llh);

    // clean up
    Py_DECREF(pre);
    Py_DECREF(post);
    Py_DECREF(rates);
    Py_DECREF(initial);
    Py_DECREF(tspan);
    Py_DECREF(times);
    Py_DECREF(events);

    return(llh_out);
}

static PyMethodDef gillespie_methods[] = {
    //...
    {"simulate",  simulate, METH_VARARGS,
     "Stochastic simulation of a mass action model for a fixed initial over a given time span"},
    {"llh",  llh, METH_VARARGS,
     "Compute the log likelihood of a given trajectory"},
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

void gillespie_fun (system_model &sys, std::vector<double> &time, std::vector<double> &events, std::vector<double> &stats, double *llh) {
    // preparations
    double t = sys.tspan[0];
    double t_max = sys.tspan[1];
    int index = 0;
    std::vector<double> state(sys.state,sys.state+sys.num_species);
    std::vector<double> propensity(sys.num_reactions);
    std::mt19937 rng(sys.seed);
    std::uniform_real_distribution<double> U;
    // sample path
    while (t < t_max) {
        // determine next event
        update_propensity(propensity,state,sys);
        bool success = next_reaction(&t,&index,sys,propensity,U(rng),U(rng),stats,llh);
        if (not success)
            break;
        // update system
        update_state(state,index,sys);
        // update output statistics
        events.push_back(index);
        time.push_back(t);
    }
    return;
}

void update_propensity(std::vector<double> &propensity, std::vector<double> &state, const system_model &sys) {
    // initialise to one
    for (unsigned i = 0; i < sys.num_reactions; i++) {
        propensity[i] = 1.0;
    }
    // calculate the stoichiometric factors
    for (unsigned i = 0; i < sys.num_reactions; i++) {
        for (unsigned j = 0; j < sys.num_species; j++) {
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

bool next_reaction (double *t, int *index, system_model &sys, std::vector<double> &propensity, double rand_1, double rand_2, std::vector<double> &stats, double *reaction_llh) {
    /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
    // calculate the reaction hazards
    std::vector<double> hazard(sys.num_reactions);
    std::vector<double> cum_hazard(sys.num_reactions);
    double total_hazard = 0.0;
    for (size_t i = 0; i < sys.num_reactions; i++ ) {
        double tmp = sys.rates[i]*propensity[i];
        total_hazard += tmp;
        hazard[i] = tmp;
        cum_hazard[i] = total_hazard;
    }
    // check if reaction happens
    bool fired = false;
    double delta_t = -std::log(rand_1)/total_hazard;
    if (total_hazard > 0.0 && *t+delta_t < sys.tspan[1]) {
        fired = true;
    }
    // perform updates
    if (fired) {
        *t += delta_t;
        // sample random event from the individual hazards
        rand_2 *= total_hazard;
        *index = 0;
        while ( cum_hazard[*index] < rand_2) {
            (*index)++;
        }
        // compute reaction llh contribution
        *reaction_llh += std::log(hazard[*index]/total_hazard);
        // update integrated stats
        for (unsigned i = 0; i < sys.num_reactions; i++) {
            stats[i] += hazard[i]*delta_t;
        }
    }
    else {
         // update integrated stats to terminal time
        for (unsigned i = 0; i < sys.num_reactions; i++) {
            stats[i] += hazard[i]*(sys.tspan[1]-*t);
        }       
    }
    return(fired);
}

void update_state(std::vector<double> &state,int index,const system_model &sys) {
    for (unsigned i = 0; i < sys.num_species; i++ ) {
        size_t ind = index*sys.num_species+i;
        state[i] += sys.S[ind];
    }
    return;
}

void update_history(std::vector<double> &state_history,std::vector<double> &state, system_model &sys) {
    for (unsigned i = 0; i < sys.num_species; i++) {
        state_history.push_back(state[i]);
    }
    return;
}

void update_total_propensity(std::vector<double> &total_propensity, const std::vector<double> &propensity, system_model &sys, double delta_t ) {
    for (unsigned i = 0; i < sys.num_reactions; i++) {
        total_propensity[i] += delta_t*propensity[i];
    }
    return;
}


void construct_trajectory(const system_model &sys,const std::vector<double> events, std::vector<double> &trajectory) {
    // get initial state
    std::vector<double> state(sys.state,sys.state+sys.num_species);
    // fill up the trajectory by iterating over the events
    for (unsigned i = 0; i < events.size(); i++ ) {
        // update the state
        int index = events[i];
        update_state(state,index,sys);
        // append to the vector
        for (unsigned j = 0; j < sys.num_species; j++) {
            trajectory[i*sys.num_species+j] = state[j];
        }
    }
}

double compute_likelihood(const system_model &sys, double *events, double *times, unsigned num_events) {
    // preparations
    std::vector<double> state(sys.state,sys.state+sys.num_species);
    double time = sys.tspan[0];
    std::vector<double> propensity(sys.num_reactions);
    std::vector<double> stats(sys.num_reactions);
    double llh = 0.0;
    // iterate over events
    for (unsigned i = 0; i < num_events; i++) {
        // compute propensity
        int index = events[i];
        update_propensity(propensity, state, sys);
        double total_propensity = 0.0;
        for (unsigned j = 0; j < sys.num_reactions; j++) {
            propensity[j] *= sys.rates[j];
            total_propensity += propensity[j];
        }
        // compute llh contributions
        llh += std::log(propensity[index]/total_propensity);
        for (unsigned j = 0; j < sys.num_reactions; j++) {
            stats[j] += (times[i]-time)*propensity[j];
        }
        //llh -= (times[i]-time)*total_propensity;
        // update
        time = times[i];
        update_state(state, index, sys);
    }
    // contribution of the final interval
    update_propensity(propensity, state, sys);
    double total_propensity = 0.0;
    for (unsigned j = 0; j < sys.num_reactions; j++) {
        stats[j] += (sys.tspan[1]-time)*propensity[j]*sys.rates[j];
    }
    for (unsigned j = 0; j < sys.num_reactions; j++) {
        llh -= stats[j];
    }
    return(llh);
}