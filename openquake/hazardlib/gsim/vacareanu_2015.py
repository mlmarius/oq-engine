# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2019 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`VacareanuEtAl2015`

"""
import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA, PGV


class VacareanuEtAl2015(GMPE):
    """

    """
    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        PGV,
        SA
    ])

    #: Supported intensity measure component is the geometric mean component
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see table 3, pages 12 - 13
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Site amplification is dependent upon Vs30
    #: For the Abrahamson et al (2013) GMPE a new term is introduced to
    #: determine whether a site is on the forearc with respect to the
    #: subduction interface, or on the backarc. This boolean is a vector
    #: containing False for a backarc site or True for a forearc or
    #: unknown site.

    REQUIRES_SITES_PARAMETERS = set(('vs30', 'backarc'))

    #: Required rupture parameters are magnitude for the interface model
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'hypo_depth'))

    #: Required distance measure is closest distance to rupture, for
    REQUIRES_DISTANCES = set(('rhypo',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type and for PGA
        C = self.COEFFS[imt]
        mean = (self._compute_magnitude_term(C, rup.mag) +
                self._compute_distance_term(C, dists) +
                self._compute_focal_depth_term(C, rup.hypo_depth) +
                self._compute_forearc_backarc_term(C, sites) +
                self._compute_site_response_term(C, sites))
        stddevs = self._get_stddevs(C, stddev_types, len(sites.vs30))
        return mean, stddevs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations as defined in Table 3
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(C['sigma'] + np.zeros(num_sites))
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(C['tau'] + np.zeros(num_sites))
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(C['phi'] + np.zeros(num_sites))

        return stddevs

    def _compute_magnitude_term(self, C, mag):
        """
        Computes the magnitude scaling term given by equation (2)
        """
        return C['c1'] + (C['c2'] * (mag - 6)) + C['c3'] * ((mag - 6) ** 2.)

    def _compute_distance_term(self, C, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return C['c4'] * np.log(dists.rhypo)

    def _compute_forearc_backarc_term(self, C, sites, dists):
        """
        Computes the forearc/backarc scaling term given by equation (4)

        """
        arc = int(sites.backarc)
        return C['c5'] * (1 - arc) * dists.rhypo + C['c6'] * arc * dists.rhypo

    def _compute_focal_depth_term(self, C, rup):
        return C['c7'] * rup.hypo_depth

    def _get_sb_sc_ss(self, sites):
        results = []
        for vs30_value in sites.vs30:
            if 180 <= vs30_value < 360:
                sb, sc, ss = 0, 1, 0
            elif 360 <= vs30_value < 800:
                sb, sc, ss = 1, 0, 0
            else:
                sc, sb, ss = 0, 0, 1
            results.append((sb, sc, ss))
        return results

    def _compute_site_response_term(self, C, sites):
        """
        Compute and return site response model term
        """
        sb_sc_ss = self._get_sb_sc_ss(sites)
        return map(
            lambda sb,sc,ss: (sb * C['c8'], sc * C['c9'], ss * C['c10']),
            sb_sc_ss
        )


    # Period-dependent coefficients (Table 3)
    COEFFS = CoeffsTable(sa_damping=5, table="""\
         imt     c1      c2      c3      c4      c5      c6      c7      c8      c9      c10     sigma   thau    phi
         pgv    10.2438  1.8264  -0.0522 -3.6280  0.0036  0.0101  0.0017  6.6201 6.9340  0       0.751   0.334   0.672
         pga     9.6231  1.4232  -0.1555 -1.1316 -0.0114 -0.0024 -0.0007 -0.0835 0.1589  0.0488  0.698   0.406   0.568
         0.1     9.6981  1.3679  -0.1423 -0.9889 -0.0135 -0.0026 -0.0017 -0.1965 0.167   0.002   0.806   0.468   0.656
         0.2     10.009  1.362   -0.1138 -1.0371 -0.0127 -0.0032 -0.0004 -0.1547 0.2861  0.086   0.792   0.469   0.638
         0.3     10.7033 1.458   -0.1187 -1.234  -0.0106 -0.0026 0       -0.1014 0.2659  0.0991  0.783   0.48    0.619
         0.4     10.7701 1.5748  -0.1439 -1.3207 -0.0093 -0.0022 0.0005  -0.1076 0.3062  0.1183  0.81    0.519   0.622
         0.5     9.2327  1.6739  -0.1664 -1.0022 -0.01   -0.0041 0.0007  -0.0259 0.2576  0.0722  0.767   0.461   0.613
         0.6     8.6445  1.7672  -0.1925 -0.8938 -0.0099 -0.0045 -0.0004 -0.1038 0.2181  0.0179  0.74    0.429   0.603
         0.7     8.7134  1.85    -0.199  -0.978  -0.0088 -0.0039 0.0002  -0.1867 0.1564  0.0006  0.735   0.426   0.599
         0.8     9.0835  1.9066  -0.2022 -1.1044 -0.0078 -0.0031 0.0005  -0.2901 0.0546  -0.1019 0.726   0.417   0.594
         0.9     9.1274  1.9662  -0.2465 -1.1437 -0.0074 -0.0031 0.0001  -0.2804 0.0884  -0.079  0.719   0.403   0.596
         1       8.9987  1.9964  -0.2658 -1.1226 -0.0071 -0.0031 -0.0009 -0.2992 0.0739  -0.0955 0.715   0.4     0.592
         1.2     8.0465  2.0432  -0.2241 -0.9654 -0.0072 -0.0041 -0.0013 -0.2681 0.1476  -0.0412 0.713   0.392   0.595
         1.4     7.0585  2.1148  -0.2167 -0.8011 -0.0078 -0.0049 -0.0013 -0.2566 0.2009  -0.0068 0.714   0.392   0.597
         1.6     6.8329  2.1668  -0.2418 -0.8036 -0.0075 -0.0047 -0.0018 -0.2268 0.2272  0.0211  0.732   0.418   0.601
         1.8     6.4292  2.1988  -0.2468 -0.7625 -0.0073 -0.0047 -0.002  -0.2464 0.22    0.0082  0.745   0.427   0.611
         2       6.3876  2.2151  -0.2289 -0.8004 -0.0066 -0.0043 -0.0024 -0.2767 0.2134  -0.0091 0.744   0.425   0.611
         2.5     4.4248  2.2541  -0.2144 -0.428  -0.0079 -0.0061 -0.0031 -0.2924 0.2108  -0.0177 0.75    0.42    0.622
         3       4.5395  2.2812  -0.2256 -0.534  -0.0072 -0.0054 -0.0034 -0.3066 0.184   -0.0387 0.765   0.436   0.629
         3.5     4.7407  2.2803  -0.2456 -0.625  -0.0065 -0.0045 -0.0041 -0.3728 0.0918  -0.1192 0.778   0.436   0.645
     """)