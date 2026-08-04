from visualization_msgs.msg import Marker, MarkerArray

from gb_optimizer.global_trajectory_publisher import GlobalRepublisher


def test_marker_array_reset_removes_old_reset_and_preserves_new_markers():
    old_reset = Marker()
    old_reset.action = Marker.DELETEALL
    first = Marker()
    first.id = 10
    first.action = Marker.ADD
    second = Marker()
    second.id = 11
    second.action = Marker.ADD

    result = GlobalRepublisher._marker_array_with_reset(
        MarkerArray(markers=[old_reset, first, second])
    )

    assert [marker.action for marker in result.markers] == [
        Marker.DELETEALL,
        Marker.ADD,
        Marker.ADD,
    ]
    assert [marker.id for marker in result.markers[1:]] == [10, 11]
