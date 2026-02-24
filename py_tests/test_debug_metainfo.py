"""
Test file for debugging metadata loading
"""


def test_all_metainfo_fixture(all_metainfo):
    """Check all_metainfo fixture return value"""
    print(f"\nall_metainfo type: {type(all_metainfo)}")
    print(f"all_metainfo keys: {list(all_metainfo.keys())}")

    sspd_metainfo = all_metainfo.get("sspd", [])
    print(f"sspd_metainfo type: {type(sspd_metainfo)}")
    print(f"sspd_metainfo length: {len(sspd_metainfo)}")
    print(f"sspd_metainfo: {sspd_metainfo}")

    euclidean_metainfo = [m for m in sspd_metainfo if m.type_d == "euclidean"]
    print(f"euclidean_metainfo length: {len(euclidean_metainfo)}")
    print(f"euclidean_metainfo: {euclidean_metainfo}")

    assert len(euclidean_metainfo) > 0, "SSPD euclidean metainfo should not be empty"


def test_sspd_euclidean_metainfo_filtering(all_metainfo):
    """Test SSPD euclidean metadata filtering"""
    sspd_metainfo = all_metainfo.get("sspd", [])

    # Check type_d field
    print("\nChecking type_d values:")
    for i, m in enumerate(sspd_metainfo):
        type_d = m.type_d
        print(f"  Entry {i}: type_d={repr(type_d)} (type={type(type_d)})")
        print(f"    type_d == 'euclidean': {type_d == 'euclidean'}")

    euclidean_metainfo = [m for m in sspd_metainfo if m.type_d == "euclidean"]
    print(f"euclidean_metainfo: {euclidean_metainfo}")

    assert len(euclidean_metainfo) > 0, "Should have euclidean entries"
