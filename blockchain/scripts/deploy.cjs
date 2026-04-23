const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const Contract = await hre.ethers.getContractFactory("DeepfakeStorage");
  const contract = await Contract.deploy();

  await contract.deployed();

  console.log("✅ Contract deployed to:", contract.address);

  // ✅ Save address automatically
  const path = require("path");

  const filePath = path.join(__dirname, "../../contract-address.json");

  fs.writeFileSync(
    filePath,
    JSON.stringify({ address: contract.address }, null, 2)
  );

  console.log("📁 Address saved to contract-address.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});