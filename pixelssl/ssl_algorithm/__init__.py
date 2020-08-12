from .ssl_null import SSLNULL
from .ssl_mt import SSLMT
from .ssl_adv import SSLADV
from .ssl_gct import SSLGCT


SSL_NULL = SSLNULL.NAME
SSL_MT = SSLMT.NAME
SSL_ADV = SSLADV.NAME
SSL_GCT = SSLGCT.NAME


SSL_ALGORITHMS = [
  SSL_NULL,
  SSL_MT,
  SSL_ADV,
  SSL_GCT,
]


del SSLNULL
del SSLMT
del SSLADV
del SSLGCT
