from easyvvuq.constants import default_campaign_prefix, Status
from easyvvuq.db.sql import CampaignDB
from easyvvuq.data_structs import CampaignInfo
from easyvvuq import Campaign
from easyvvuq.analysis import SCAnalysis
from easyvvuq.encoders import GenericEncoder
import tempfile
import os
import numpy as np


class CustomCampaign(Campaign):
    # ----------------------------------------------------------------------
    # changes :
    # send runs_dir='SWEEP' when we call CampaignInfo
    # change location of campaign.db to work directory
    # ----------------------------------------------------------------------

    def init_fresh(self, name, db_type='sql',
                   db_location=None, work_dir='.'):

        # Create temp dir for campaign
        campaign_prefix = default_campaign_prefix
        if name is not None:
            campaign_prefix = name

        campaign_dir = tempfile.mkdtemp(prefix=campaign_prefix, dir=work_dir)

        self._campaign_dir = os.path.relpath(campaign_dir, start=work_dir)

        self.db_location = db_location
        self.db_type = db_type

        if self.db_type == 'sql':
            from easyvvuq.db.sql import CampaignDB
            if self.db_location is None:
                self.db_location = "sqlite:///" + work_dir + "/campaign.db"
                # self.db_location = "sqlite:///" + self.campaign_dir + "/campaign.db"
        else:
            message = (f"Invalid 'db_type' {db_type}. Supported types are "
                       f"'sql'.")
            logger.critical(message)
            raise RuntimeError(message)
        from easyvvuq import __version__
        info = CampaignInfo(
            name=name,
            campaign_dir_prefix=default_campaign_prefix,
            easyvvuq_version=__version__,
            campaign_dir=self.campaign_dir,
            #runs_dir=os.path.join(campaign_dir, 'runs')
            runs_dir=os.path.join(campaign_dir, 'SWEEP')
        )
        self.campaign_db = CampaignDB(location=self.db_location,
                                      new_campaign=True,
                                      name=name, info=info)

        # Record the campaign's name and its associated ID in the database
        self.campaign_name = name
        self.campaign_id = self.campaign_db.get_campaign_id(self.campaign_name)

    # ----------------------------------------------------------------------
    # changes :
    # return generated run_ids when we call populate_runs_dir
    # ----------------------------------------------------------------------

    def populate_runs_dir(self):

        # Get the encoder for this app. If none is set, only the directory structure
        # will be created.
        active_encoder = self._active_app_encoder
        if active_encoder is None:
            logger.warning(
                'No encoder set for this app. Creating directory structure only.')

        run_ids = []

        for run_id, run_data in self.campaign_db.runs(
                status=Status.NEW, app_id=self._active_app['id']):

            # Make directory for this run's output
            os.makedirs(run_data['run_dir'])

            # Encode run
            if active_encoder is not None:
                active_encoder.encode(params=run_data['params'],
                                      target_dir=run_data['run_dir'])

            run_ids.append(run_id)
        self.campaign_db.set_run_statuses(run_ids, Status.ENCODED)
        return run_ids


class CustomSCAnalysis(SCAnalysis):

    # ----------------------------------------------------------------------
    # changes :
    # add file input parameter to save generated plot
    # ----------------------------------------------------------------------

    def adaptation_histogram(self, file=None):
        """
        Parameters
        ----------
        None

        Returns
        -------
        Plots a bar chart of the maximum order of the quadrature rule
        that is used in each dimension. Use in case of the dimension adaptive
        sampler to get an idea of which parameters were more refined than others.
        This gives only a first-order idea, as it only plots the max quad
        order independently per input parameter, so higher-order refinements
        that were made do not show up in the bar chart.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[4, 8])
        ax = fig.add_subplot(111, ylabel='max quadrature order',
                             title='Number of refinements = %d'
                             % self.sampler.number_of_adaptations)
        # find max quad order for every parameter
        adapt_measure = np.max(self.l_norm, axis=0)
        ax.bar(range(adapt_measure.size), height=adapt_measure)
        params = list(self.sampler.vary.get_keys())
        ax.set_xticks(range(adapt_measure.size))
        ax.set_xticklabels(params)
        plt.xticks(rotation=90)
        plt.tight_layout()

        if file == None:
            plt.show()
        else:
            plt.savefig(file, dpi=400)


class CustomEncoder(GenericEncoder, encoder_name='CustomEncoder'):

    def encode(self, params={}, target_dir=''):
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
        curve = default_mortality * params["mortality_factor"]
        params["mortality_curve"] = curve

        proportion_symptomatic = [params["p_symptomatic"]] * 17
        params["Proportion_symptomatic"] = proportion_symptomatic

        super().encode(params, target_dir)
