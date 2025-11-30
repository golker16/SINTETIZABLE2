"""
Stub de sklearn.decomposition. Si alguien intenta usarlo, lanzamos error claro.
"""
class _MissingScikitLearn(RuntimeError):
    pass

def __getattr__(name):
    raise _MissingScikitLearn(
        "Esta app no incluye scikit-learn. "
        f"Se intentó acceder a sklearn.decomposition.{name}. "
        "Instala scikit-learn si realmente necesitas funciones de descomposición."
    )
