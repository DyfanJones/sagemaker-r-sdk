# Position is a tuple that includes the last read timestamp and the number of items that were read
# at that time. This is used to figure out which event to start with on the next read.
sm_env <- new.env(parent=emptyenv())
sm_env$positions <- NULL
