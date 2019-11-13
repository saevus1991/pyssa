#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <cmath>

void plus(double *arg1, double* arg2, unsigned n);

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
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC
PyInit_spam(void)
{
    return PyModule_Create(&spammodule);
}

int main() {

    // create a c array
    unsigned n = 10;
    double a[10];
    for (int i = 0; i < n; i++) {
        a[i] = i*i;
        std::cout << a[i] << std::endl;
    }

}

void plus(double *arg1, double* arg2, double *output, unsigned n) {
    for (int i = 0; i < n; i++) {
        std::cout << *arg1 << std::endl;
    }
}