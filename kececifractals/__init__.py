# __init__.py
# Bu dosya paketin başlangıç noktası olarak çalışır.
# Alt modülleri yükler, sürüm bilgileri tanımlar ve geriye dönük uyumluluk için uyarılar sağlar.

from __future__ import annotations
import importlib
import os
import warnings

# if os.getenv("DEVELOPMENT") == "true":
    # importlib.reload(kececifractals) # F821 undefined name 'kececifractals'

# Paket sürüm numarası
__version__ = "0.1.4"

__all__ = [
    'random_soft_color',
    '_draw_circle_patch',
    '_draw_recursive_circles',
    'kececifractals_circle',
    '_draw_recursive_qec',
    'visualize_qec_fractal',
    '_draw_recursive_stratum_circles',
    'visualize_stratum_model',
    'visualize_sequential_spectrum'
]

# Göreli modül içe aktarmaları
# F401 hatasını önlemek için sadece kullanacağınız şeyleri dışa aktarın
# Aksi halde linter'lar "imported but unused" uyarısı verir
try:
    #from .kececifractals import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    #from . import kececifractals  # Modülün kendisine doğrudan erişim isteniyorsa
    from .kececifractals import (
        random_soft_color, 
        _draw_circle_patch, 
        _draw_recursive_circles, 
        kececifractals_circle,  
        _draw_recursive_qec, 
        visualize_qec_fractal, 
        _draw_recursive_stratum_circles, 
        visualize_stratum_model, 
        visualize_sequential_spectrum
    )
except ImportError as e:
    warnings.warn(f"Gerekli modül yüklenemedi: {e}", ImportWarning)

# Eski bir fonksiyonun yer tutucusu - gelecekte kaldırılacak
def eski_fonksiyon():
    """
    Kaldırılması planlanan eski bir fonksiyondur.
    Lütfen alternatif fonksiyonları kullanın.
    """
    warnings.warn(
        "eski_fonksiyon() artık kullanılmamaktadır ve gelecekte kaldırılacaktır. "
        "Lütfen yeni alternatif fonksiyonları kullanın. "
        "Keçeci Fractals; Python 3.9-3.14 sürümlerinde sorunsuz çalışmalıdır.",
        category=DeprecationWarning,
        stacklevel=2
    )
