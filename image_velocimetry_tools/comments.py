class Comments:
    """Comment object

    Attributes:
    -----------
    comments: dict
        Dictionary of comments.

    """

    def __init__(self):
        self.comments = {
            "Video Preprocessing": [],
            "Image Frame Processing": [],
            "Orthorectification": [],
            "Cross-Section Geometry": [],
            "Grid Preparation": [],
            "Space-Time Image Velocimetry (Exhaustive)": [],
            "Space-Time Image Results": [],
            "Discharge": [],
            "Reporting": [],
            "System": [],
            "Other": [],
        }

    def load_dict(self, comment_dict):
        """Loads data from session file from comment dictionary format.

        Parameters:
        -----------
            ac_dict: dict

        """
        keys = self.comments.keys()

        for key in keys:
            data = comment_dict[key]
            if isinstance(data, str):
                self.comments[key].append(data)
            if isinstance(data, list):
                self.comments[key] = data

    def append_comment(self, key, comment):
        """Adds comment to key.

        Parameters:
        -----------
            comment: lst
            key: key to append string

        """

        self.comments[key].append(comment)
