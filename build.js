const fs = require("fs-extra");
const path = require("path");
const glob = require("glob");
const handlebars = require("handlebars");

const srcDir = "src";
const distDir = "docs";

// Clean dist folder
fs.emptyDirSync(distDir);

// Register partials
const partialsDir = path.join(srcDir, "views/partials");
glob.sync("**/*.hbs", { cwd: partialsDir }).forEach(file => {
  const name = path.basename(file, ".hbs");
  const template = fs.readFileSync(path.join(partialsDir, file), "utf8");
  handlebars.registerPartial(name, template);
});

// Register layouts as partials (for layout inheritance)
const layoutsDir = path.join(srcDir, "views/layouts");
const layouts = {};
glob.sync("**/*.hbs", { cwd: layoutsDir }).forEach(file => {
  const name = path.basename(file, ".hbs");
  const template = fs.readFileSync(path.join(layoutsDir, file), "utf8");
  layouts[name] = handlebars.compile(template);
  handlebars.registerPartial(name, template);
});

// Compile each page
const pagesDir = path.join(srcDir, "views");
glob.sync("*.hbs", { cwd: pagesDir }).forEach(file => {
  const templateContent = fs.readFileSync(path.join(pagesDir, file), "utf8");
  
  // Check if template uses layout inheritance
  const layoutMatch = templateContent.match(/{{#>\s*(\w+)/);
  if (layoutMatch) {
    const layoutName = layoutMatch[1];
    if (layouts[layoutName]) {
      // Extract the content between layout tags
      const contentMatch = templateContent.match(/{{#>\s*\w+[^}]*}}([\s\S]*?){{\/\w+}}/);
      const titleMatch = templateContent.match(/title="([^"]+)"/);
      
      if (contentMatch) {
        const content = contentMatch[1].trim();
        const title = titleMatch ? titleMatch[1] : 'Page';
        
        const html = layouts[layoutName]({
          title: title,
          body: content
        });
        
        const outPath = path.join(distDir, file.replace(".hbs", ".html"));
        fs.outputFileSync(outPath, html);
        console.log(`Built ${outPath} with ${layoutName} layout`);
        return;
      }
    }
  }
  
  // Fallback: compile as regular template
  const compiled = handlebars.compile(templateContent);
  const html = compiled({});
  const outPath = path.join(distDir, file.replace(".hbs", ".html"));
  fs.outputFileSync(outPath, html);
  console.log(`Built ${outPath}`);
});

// Copy CSS
fs.copySync(path.join(srcDir, "style.css"), path.join(distDir, "style.css"));
console.log("Copied style.css");

// Copy images
const imgSrcDir = path.join(srcDir, "views/img");
const imgDistDir = path.join(distDir, "img");
if (fs.existsSync(imgSrcDir)) {
  fs.copySync(imgSrcDir, imgDistDir);
  console.log("Copied images to docs/img");
}