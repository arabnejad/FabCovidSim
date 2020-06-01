import numpy as np
import easyvvuq as uq

class CustomEncoder(uq.encoders.JinjaEncoder, encoder_name='CustomEncoder'):
    def encode(self, params={}, target_dir='', fixtures=None):
        """
        # Logistic curve for mortality
        k = params["mortality_k"]
        x0 = params["mortality_x0"]

        age = np.arange(5,90,5)
        curve = 1 / (1 + e**(-k*(age-x0)))
        """
        # scale default values found in pre param file
        default_mortality = np.array([0,
                                      1.60649128,
                                      2.291051747,
                                      2.860938008,
                                      3.382077741,
                                      3.880425012,
                                      4.37026577,
                                      4.861330415,
                                      5.361460943,
                                      5.877935626,
                                      6.4183471,
                                      6.991401405,
                                      7.607881726,
                                      8.282065409,
                                      9.034104744,
                                      9.894486491,
                                      10.91341144,
                                      12.18372915,
                                      13.9113346,
                                      16.74394356,
                                      22.96541429])
        curve = default_mortality * params["Mortality_factor"]
        params["Mortality_curve"] = curve

        proportion_symptomatic = [params["Proportion_symptomatic"]] * 17
        params["Proportion_symptomatic_array"] = proportion_symptomatic

        default_contact_rates = np.array([0.6, 0.7, 0.75,  1, 1, 1, 1, 1, 1, 1, 
                                          1, 1, 1, 1, 1, 0.75,  0.5])
        contact_rates = default_contact_rates ** params["Relative_spatial_contact_rates_by_age_power"]
        params["Relative_spatial_contact_rates_by_age_array"] = contact_rates

        super().encode(params, target_dir, fixtures)


