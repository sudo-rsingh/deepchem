const SmilesDrawer = require("smiles-drawer");

//import SmilesDrawer from "smiles-drawer"

let smilesDrawer = new SmilesDrawer.Drawer({ width: 250, height: 250 });

SmilesDrawer.parse('C1CCCCC1', function (tree) {
    smilesDrawer.draw(tree, 'output-canvas', 'light', false);
}, function (err) {
    console.log(err);
})