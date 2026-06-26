"""
conftest.py — project-wide pytest configuration

Applies the IE University SSL proxy fix before any test module is imported.
The university network uses an SSL inspection proxy that replaces server
certificates. Without this patch, any test that makes an HTTPS call to
Google APIs fails with CERTIFICATE_VERIFY_FAILED.
"""
import httpx

_orig_init = httpx.Client.__init__


def _ssl_fixed_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_init(self, *args, **kwargs)


httpx.Client.__init__ = _ssl_fixed_init
