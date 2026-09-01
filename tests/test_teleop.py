import numpy as np

from yamkit.teleop import TeleopSession


def test_teleop_engage_and_track(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, auto_engage=True)
    leader = fake_connect["left_leader"]
    follower = fake_connect["left_follower"]
    leader.pos = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    leader.encoder[0].position = 0.5  # trigger half squeezed -> gripper 0.5
    stats = session.run(duration=0.1)
    assert stats.ticks > 5
    assert np.allclose(follower.pos[:6], leader.pos, atol=0.05)
    assert follower.pos[6] == np.isclose(follower.pos[6], 0.5) or abs(follower.pos[6] - 0.5) < 0.1
    assert leader.closed and follower.closed


def test_button_toggles_engage(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["right_follower"], hz=500.0, sync_seconds=0.01)
    pair = session.pairs[0]
    leader = fake_connect["right_leader"]
    assert not pair.engaged
    session.step()
    leader.encoder[0].io_inputs = [1, 0]  # press
    session.step()
    assert pair.engaged
    session.step()  # held: no toggle
    assert pair.engaged
    leader.encoder[0].io_inputs = [0, 0]
    session.step()
    leader.encoder[0].io_inputs = [1, 0]  # press again
    session.step()
    assert not pair.engaged
    session.shutdown()
