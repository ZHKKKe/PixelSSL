from .ssl_null import SSLNULL
from .ssl_mt import SSLMT
from .ssl_adv import SSLADV
from .ssl_s4l import SSLS4L
from .ssl_gct import SSLGCT
from .ssl_cct import SSLCCT


SSL_NULL = SSLNULL.NAME
SSL_MT = SSLMT.NAME
SSL_ADV = SSLADV.NAME
SSL_S4L = SSLS4L.NAME
SSL_GCT = SSLGCT.NAME
SSL_CCT = SSLCCT.NAME


SSL_ALGORITHMS = [
  SSL_NULL,
  SSL_MT,
  SSL_ADV,
  SSL_S4L,
  SSL_GCT,
  SSL_CCT,
]


del SSLNULL
del SSLMT
del SSLADV
del SSLS4L
del SSLGCT
del SSLCCT
