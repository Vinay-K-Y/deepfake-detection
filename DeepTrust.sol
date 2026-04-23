// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DeepTrust {

    struct Result {
        string fileHash;
        string verdict;
        uint confidence;
        uint timestamp;
    }

    Result[] public results;

    function storeResult(
        string memory _hash,
        string memory _verdict,
        uint _confidence
    ) public {
        results.push(Result(_hash, _verdict, _confidence, block.timestamp));
    }

    function getResult(uint index) public view returns (
        string memory, string memory, uint, uint
    ) {
        Result memory r = results[index];
        return (r.fileHash, r.verdict, r.confidence, r.timestamp);
    }
}