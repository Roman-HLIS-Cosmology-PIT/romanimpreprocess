"""
Process logging tools.

Classes
-------
ProcessLog
    This logs events that occurred in the processing.

"""


class ProcessLog:
    """
    This logs events that occurred in the processing.

    This can be initialized as ``mylog = ProcessLog()``.
    Then other methods can be used to add to the log.

    Parameters
    ----------
    None

    Attributes
    ----------
    output : str
        Collected output.
    reffiles : dict
        Reference files used.

    Methods
    -------
    append
        Add more output.

    """

    def __init__(self):
        self.output = ""
        self.reffiles = {}

    def append(self, newoutput):
        """
        Add more output.

        Parameters
        ----------
        newoutput : str
            The output to append.

        Returns
        -------
        None

        """

        self.output += newoutput
