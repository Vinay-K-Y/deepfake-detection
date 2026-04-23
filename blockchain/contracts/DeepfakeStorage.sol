// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DeepfakeStorage {

    struct Record {
        string fileHash;
        string result;
        uint256 timestamp;
    }

    Record[] public records;

    function storeResult(string memory _hash, string memory _result) public {
        records.push(Record(_hash, _result, block.timestamp));
    }

    function getRecord(uint256 index) public view returns (
        string memory,
        string memory,
        uint256
    ) {
        Record memory r = records[index];
        return (r.fileHash, r.result, r.timestamp);
    }

    // ✅ NEW (IMPORTANT)
    function getTotalRecords() public view returns (uint256) {
        return records.length;
    }
}