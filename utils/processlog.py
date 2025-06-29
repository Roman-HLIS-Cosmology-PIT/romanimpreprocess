class ProcessLog:

    """This logs events that occurred in the processing.

    Attributes:
    output : string containing output information
    reffiles : dictionary of reference files used

    This can be initialized as mylog = ProcessLog().
    Then other methods can be used to add to the log.
    """

    def __init__(self):
        self.output = ""
        self.reffiles = {}

    def append(self, newoutput):
        """newoutput should be a string"""
        self.output += newoutput
