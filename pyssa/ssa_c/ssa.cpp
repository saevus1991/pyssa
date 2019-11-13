#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <cmath>

void plus(double *a, double* b, double *c, unsigned n);

static PyObject *array_plus(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyMethodDef SpamMethods[] = {
    //...
    {"system",  array_plus, METH_VARARGS,
     "Execute a shell command."},
    //...ı
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    "spam_doc", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC
PyInit_spam(void)
{
    return PyModule_Create(&spammodule);
}

int main() {

    // init arrays
    unsigned n = 10;
    double *a;
    a = (double *) malloc(n*sizeof(double));
    double *b;
    b = (double *) malloc(n*sizeof(double));
    double *c;
    c = (double *) malloc(n*sizeof(double));

    // test numpy array
    numpyArray<double,3> a

    // fill a and b
    for (int i = 0; i < n; i++) {
        a[i] = i*i;
        b[i] = 100-i;
    }

    // add both arrays
    plus(a, b, c, n);

    // print result
    for (int i = 0; i < n; i++) {
        std::cout << c[i] << std::endl;
    }

    free(a);
    free(b);
    free(c);

}

void plus(double *a, double* b, double *c, unsigned n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}