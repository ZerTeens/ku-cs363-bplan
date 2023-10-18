class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.state = -1
        self.mode = -1
        self.title = ""
        self.startTime = ""
        self.endTime = ""
        self.description = ""

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def startTime(self):
        return self._startTime

    @startTime.setter
    def startTime(self, value):
        self._startTime = value

    @property
    def endTime(self):
        return self._endTime

    @endTime.setter
    def endTime(self, value):
        self._endTime = value

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value
