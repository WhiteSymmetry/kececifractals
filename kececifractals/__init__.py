# __init__.py
"""
Keçeci Fractals - 2D ve 3D Fraktal Oluşturma Kütüphanesi

Bu kütüphane, Keçeci fraktallarını oluşturmak için kapsamlı bir araç seti sağlar.
2D dairesel fraktallar, kuantum hata düzeltme görselleştirmeleri ve 3D fraktallar içerir.

Python: 3.11-3.15
"""

from __future__ import annotations
import warnings
from typing import List, Optional, Tuple, Union, Callable

# Paket sürüm numarası
__version__ = "0.2.1"
__author__ = "Mehmet Keçeci"
__email__ = "mkececi@yaani.com"
__license__ = "AGPL3.0-or-later"

# Ana fonksiyonları dışa aktar
__all__ = [
    # Genel yardımcı fonksiyonlar
    "random_soft_color",
    "_parse_color",
    # 2D Fraktallar
    "kececifractals_circle",
    # Kuantum Hata Düzeltme
    "visualize_qec_fractal",
    # 3D Fraktallar
    "_generate_recursive_3d_fractal",
    "generate_color_function",
    "get_cmap_safe",
    "draw_3d_sphere",
    "kececi_3d_fractal",
    "kececifractals_3d",
    "optimized_3d_fractal",
    "generate_single_fractal",
    "generate_fractal_directly",
    "generate_simple_3d_fractal",
    # Stratum Modeli
    "visualize_stratum_model",
    "visualize_sequential_spectrum",
    # Yardımcı fonksiyonlar (internal)
    "_draw_circle_patch",
    "_draw_recursive_circles",
    "_draw_recursive_qec",
    "_draw_recursive_stratum_circles",
    # Örnek fonksiyonlar (isteğe bağlı)
    "example_multiple_fractals",
    "example_view_angles",
    "example_simple_fractal",
    # Exception'lar
    "KececiFractalError",
    "FractalParameterError",
    "ColorParseError",
    "ThreeDNotSupportedError",
    "InvalidAxisError",
    "draw_kececi_spiral",
    "draw_qec_vortex",
    "draw_chaotic_shells",
    "draw_sphere",
    "get_icosahedron_vertices",
]

try:
    # from .kececifractals import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    # from . import kececifractals  # Modülün kendisine doğrudan erişim isteniyorsa
    from .kececifractals import (
        random_soft_color,
        _draw_circle_patch,
        _draw_recursive_circles,
        kececifractals_circle,
        _draw_recursive_qec,
        visualize_qec_fractal,
        _draw_recursive_stratum_circles,
        visualize_stratum_model,
        visualize_sequential_spectrum,
        generate_color_function,
        get_cmap_safe,
        _parse_color,
        _generate_recursive_3d_fractal,
        draw_3d_sphere,
        kececi_3d_fractal,
        kececifractals_3d,
        optimized_3d_fractal,
        generate_single_fractal,
        generate_fractal_directly,
        generate_simple_3d_fractal,
        # Örnek fonksiyonlar (isteğe bağlı)
        example_multiple_fractals,
        example_view_angles,
        example_simple_fractal,
        # Exception'lar
        KececiFractalError,
        FractalParameterError,
        ColorParseError,
        ThreeDNotSupportedError,
        InvalidAxisError,
        draw_kececi_spiral,
        draw_qec_vortex,
        draw_chaotic_shells,
        draw_sphere,
        get_icosahedron_vertices,
    )
except ImportError as e:
    warnings.warn(f"Gerekli modül yüklenemedi: {e}", ImportWarning)


# Versiyon kontrolü
def check_python_version():
    """Python sürümünü kontrol et."""
    import sys

    python_version = sys.version_info

    if python_version < (3, 11):
        warnings.warn(
            f"Keçeci Fractals Python 3.11+ için tasarlanmıştır. "
            f"Mevcut sürüm: {python_version.major}.{python_version.minor}. "
            f"Beklenmeyen hatalarla karşılaşabilirsiniz.",
            RuntimeWarning,
        )

    # Gelecekteki sürüm uyumluluğu
    if python_version >= (3, 16):
        warnings.warn(
            f"Python {python_version.major}.{python_version.minor} henüz tam olarak test edilmemiştir. "
            f"Uyumluluk sorunları olabilir.",
            FutureWarning,
        )


# Python sürümünü kontrol et (ancak import sırasında değil)
try:
    check_python_version()
except:
    pass  # İlk yükleme sırasında hata verme


# Modülleri güvenli şekilde import et
def _safe_import():
    """Modülleri güvenli bir şekilde import eder."""
    try:
        # Ana modülü import et
        # from . import kececifractals

        # Genel fonksiyonlar
        from .kececifractals import (
            random_soft_color,
            _parse_color,
            _draw_circle_patch,
            _draw_recursive_circles,
            kececifractals_circle,
            _draw_recursive_qec,
            visualize_qec_fractal,
            _draw_recursive_stratum_circles,
            visualize_stratum_model,
            visualize_sequential_spectrum,
            # 3D fonksiyonları kontrol et
            _generate_recursive_3d_fractal,
            generate_color_function,
            get_cmap_safe,
            draw_3d_sphere,
            kececi_3d_fractal,
            kececifractals_3d,
            optimized_3d_fractal,
            generate_single_fractal,
            generate_fractal_directly,
            generate_simple_3d_fractal,
            # Örnek fonksiyonlar (isteğe bağlı)
            example_multiple_fractals,
            example_view_angles,
            example_simple_fractal,
            # Exception'lar
            KececiFractalError,
            FractalParameterError,
            ColorParseError,
            ThreeDNotSupportedError,
            InvalidAxisError,
            draw_sphere,
            get_icosahedron_vertices,
        )

    except ImportError as e:
        warnings.warn(
            f"Keçeci Fractals modülü yüklenemedi: {e}\n"
            f"Lütfen kurulumun doğru olduğundan emin olun.",
            ImportWarning,
            stacklevel=2,
        )
        return False


# Kullanım kolaylığı için alias'lar
kf_circle = kececifractals_circle
kf_3d = kececi_3d_fractal
qec_viz = visualize_qec_fractal

# Geriye dönük uyumluluk için alias'ları __all__'a ekle
__all__.extend(["kf_circle", "kf_3d", "qec_viz"])


# Paket hakkında bilgi
def about():
    """Paket hakkında bilgi gösterir."""
    info = f"""
    Keçeci Fractals v{__version__}
    {'=' * 40}
    
    Özellikler:
    • 2D dairesel fraktallar
    • Kuantum Hata Düzeltme görselleştirmeleri
    • 3D fraktallar
    • Stratum modeli görselleştirmeleri
    
    Python sürümü: 3.11-3.15
    Lisans: {__license__}
    """
    print(info)


# Paket kullanım istatistikleri (opsiyonel)
class _UsageTracker:
    """Kullanım istatistiklerini takip eder (anonim)."""

    _instance_count = 0

    @classmethod
    def track_usage(cls, function_name):
        """Fonksiyon kullanımını takip eder."""
        cls._instance_count += 1
        # Burada isteğe bağlı olarak logging yapılabilir
        # Ancak gizlilik için varsayılan olarak kapalı


# Eski fonksiyonlar için deprecation uyarıları
def _deprecated_function(old_name, new_name, version):
    """Eski fonksiyonlar için uyarı oluşturur."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{old_name}()' artık kullanılmamaktadır ve v{version}'da kaldırılacaktır. "
                f"Lütfen '{new_name}()' kullanın.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Eski fonksiyon örneği (dokümantasyon için)
@_deprecated_function("eski_fraktal", "kececifractals_circle", "0.3.0")
def eski_fraktal():
    """Eski fraktal fonksiyonu - kullanmayın."""
    return kececifractals_circle()


# Test fonksiyonu
def test_all():
    """Tüm fonksiyonları test et (geliştirici kullanımı)."""
    try:
        print("Keçeci Fractals Test Başlatılıyor...")
        print(f"Sürüm: {__version__}")

        # Test import
        from . import kececifractals

        print("✓ Ana modül yüklendi")

        # Test fonksiyonları
        color = random_soft_color()
        print(f"✓ Random color: {color}")

        # 2D test
        print("✓ 2D fonksiyonlar mevcut")

        # 3D test
        try:
            from mpl_toolkits.mplot3d import Axes3D

            print("✓ 3D desteği mevcut")
        except:
            print("✗ 3D desteği yok")

        print("\nTest başarıyla tamamlandı!")
        return True

    except Exception as e:
        print(f"\nTest başarısız: {e}")
        return False


# Geliştirici modu kontrolü
def is_development_mode():
    """Geliştirici modunda olup olmadığını kontrol eder."""
    import os

    return os.getenv("KECECI_DEVELOPMENT") == "1" or os.getenv("DEVELOPMENT") == "true"


# Paket yüklendiğinde mesaj göster (geliştirici modunda)
if is_development_mode():
    print(f"Keçeci Fractals v{__version__} (geliştirici modu) yüklendi")

    # Ek kontrol
    try:
        check_python_version()
    except:
        pass

# Paket metadata
package_info = {
    "name": "kececifractals",
    "version": __version__,
    "description": "2D ve 3D Keçeci fraktalları oluşturma kütüphanesi",
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.11",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: AGPL3.0-or-later",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Graphics",
    ],
}


# Özel exception'lar
class KececiFractalError(Exception):
    """Keçeci Fractals için temel exception."""

    pass


class FractalParameterError(KececiFractalError):
    """Fraktal parametre hatası."""

    pass


class ColorParseError(KececiFractalError):
    """Renk parse hatası."""

    pass


class ThreeDNotSupportedError(KececiFractalError):
    """3D desteklenmiyor hatası."""

    pass


class InvalidAxisError(KececiFractalError):
    """Geçersiz eksen hatası."""

    pass


# Exception'ları __all__'a ekle
__all__.extend(
    [
        "KececiFractalError",
        "FractalParameterError",
        "ColorParseError",
        "ThreeDNotSupportedError",
    ]
)

# Paket yüklendiğinde otomatik olarak about() çağrılmasın
# Kullanıcı isterse çağırabilir

# Son kontrol: eğer main modülü isek test çalıştır
if __name__ == "__main__":
    print("Keçeci Fractals __init__.py çalıştırılıyor...")
    about()
    print("\nHızlı test:")
    test_all()
