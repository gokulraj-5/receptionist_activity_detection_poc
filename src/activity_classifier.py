class ActivityClassifier:

    def __init__(self, active_confirmation=1.0, away_confirmation=2.0):

        self.active_confirmation = active_confirmation
        self.away_confirmation = away_confirmation

        self.current_status = "UNKNOWN"

        self.active_start_time = None
        self.away_start_time = None

    def update(self, people_inside, current_time):

        # Case 1: At least one person is inside ROI
        if people_inside > 0:

            self.away_start_time = None

            if self.current_status != "ACTIVE":

                if self.active_start_time is None:
                    self.active_start_time = current_time

                elapsed = current_time - self.active_start_time

                if elapsed >= self.active_confirmation:
                    self.current_status = "ACTIVE"

            else:
                self.active_start_time = None

        # Case 2: Nobody is inside ROI
        else:

            self.active_start_time = None

            if self.current_status == "ACTIVE":

                if self.away_start_time is None:
                    self.away_start_time = current_time

                elapsed = current_time - self.away_start_time

                if elapsed >= self.away_confirmation:
                    self.current_status = "AWAY"

            elif self.current_status == "UNKNOWN":

                if self.away_start_time is None:
                    self.away_start_time = current_time

                elapsed = current_time - self.away_start_time

                if elapsed >= self.away_confirmation:
                    self.current_status = "AWAY"

        return self.current_status