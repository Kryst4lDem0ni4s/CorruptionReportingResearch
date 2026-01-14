/**
 * ESLint Configuration for Corruption Reporting Frontend
 * Version: 1.0.0
 * Description: Code style and quality rules for JavaScript/Node.js
 * 
 * This configuration provides:
 * - Node.js environment settings
 * - ES6+ syntax support
 * - Code quality rules
 * - Best practices enforcement
 * - Security checks
 * 
 * Dependencies: None (uses ESLint built-in rules)
 * Optional: eslint (dev dependency)
 * 
 * Usage:
 * npm install --save-dev eslint
 * npx eslint .
 * npx eslint --fix .
 */

module.exports = {
    // ============================================
    // ENVIRONMENT
    // ============================================
    
    env: {
        node: true,           // Node.js global variables and scoping
        es2021: true,         // ES2021 globals
        commonjs: true        // CommonJS global variables and scoping
    },
    
    // ============================================
    // PARSER OPTIONS
    // ============================================
    
    parserOptions: {
        ecmaVersion: 2021,    // ECMAScript version
        sourceType: 'module'  // Allow ES6 imports
    },
    
    // ============================================
    // EXTENDS
    // ============================================
    
    extends: [
        'eslint:recommended'  // Use ESLint recommended rules
    ],
    
    // ============================================
    // GLOBAL VARIABLES
    // ============================================
    
    globals: {
        process: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        module: 'readonly',
        require: 'readonly',
        exports: 'readonly',
        console: 'readonly',
        Buffer: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        setImmediate: 'readonly',
        clearImmediate: 'readonly'
    },
    
    // ============================================
    // RULES
    // ============================================
    
    rules: {
        // ========================================
        // POSSIBLE ERRORS
        // ========================================
        
        'no-console': 'off',                    // Allow console (needed for logging)
        'no-debugger': 'warn',                  // Warn on debugger statements
        'no-extra-parens': 'off',               // Allow extra parentheses
        'no-template-curly-in-string': 'error', // Detect invalid template strings
        'no-unsafe-optional-chaining': 'error', // Prevent unsafe optional chaining
        
        // ========================================
        // BEST PRACTICES
        // ========================================
        
        'curly': ['error', 'all'],              // Require curly braces
        'default-case': 'warn',                 // Require default in switch
        'dot-notation': 'warn',                 // Enforce dot notation
        'eqeqeq': ['error', 'always'],          // Require === and !==
        'no-alert': 'warn',                     // Discourage alert/confirm/prompt
        'no-caller': 'error',                   // Disallow arguments.caller/callee
        'no-eval': 'error',                     // Disallow eval()
        'no-extend-native': 'error',            // Disallow extending native objects
        'no-extra-bind': 'warn',                // Disallow unnecessary bind()
        'no-implicit-globals': 'error',         // Disallow global variables
        'no-implied-eval': 'error',             // Disallow implied eval()
        'no-lone-blocks': 'warn',               // Disallow unnecessary blocks
        'no-loop-func': 'warn',                 // Disallow functions in loops
        'no-multi-spaces': 'warn',              // Disallow multiple spaces
        'no-new': 'warn',                       // Disallow new without assignment
        'no-new-func': 'error',                 // Disallow new Function()
        'no-new-wrappers': 'error',             // Disallow new String/Number/Boolean
        'no-param-reassign': 'warn',            // Disallow parameter reassignment
        'no-return-assign': 'error',            // Disallow assignment in return
        'no-self-compare': 'error',             // Disallow self comparison
        'no-sequences': 'error',                // Disallow comma operator
        'no-throw-literal': 'error',            // Require throwing Error objects
        'no-unused-expressions': 'warn',        // Disallow unused expressions
        'no-useless-concat': 'warn',            // Disallow unnecessary concat
        'no-useless-return': 'warn',            // Disallow unnecessary return
        'no-void': 'error',                     // Disallow void operator
        'no-with': 'error',                     // Disallow with statements
        'prefer-promise-reject-errors': 'error', // Require Error objects in Promise.reject
        'require-await': 'warn',                // Disallow async without await
        'yoda': 'warn',                         // Disallow Yoda conditions
        
        // ========================================
        // VARIABLES
        // ========================================
        
        'no-shadow': 'warn',                    // Disallow variable shadowing
        'no-undef': 'error',                    // Disallow undeclared variables
        'no-undef-init': 'warn',                // Disallow initializing to undefined
        'no-unused-vars': ['warn', {            // Warn on unused variables
            'argsIgnorePattern': '^_',          // Ignore args starting with _
            'varsIgnorePattern': '^_'           // Ignore vars starting with _
        }],
        'no-use-before-define': ['error', {     // Disallow use before definition
            'functions': false,                 // Allow function hoisting
            'classes': true,
            'variables': true
        }],
        
        // ========================================
        // STYLE
        // ========================================
        
        'array-bracket-spacing': ['warn', 'never'],        // No spaces in brackets
        'block-spacing': 'warn',                           // Require space in blocks
        'brace-style': ['warn', '1tbs', {                  // 1TBS brace style
            'allowSingleLine': true
        }],
        'camelcase': ['warn', {                            // Require camelCase
            'properties': 'never'
        }],
        'comma-dangle': ['warn', 'never'],                 // No trailing commas
        'comma-spacing': 'warn',                           // Space after comma
        'comma-style': 'warn',                             // Comma at end of line
        'computed-property-spacing': 'warn',               // No spaces in computed properties
        'eol-last': 'warn',                                // Newline at end of file
        'func-call-spacing': 'warn',                       // No space before ()
        'indent': ['warn', 4, {                            // 4-space indentation
            'SwitchCase': 1
        }],
        'key-spacing': 'warn',                             // Space after colon
        'keyword-spacing': 'warn',                         // Space around keywords
        'linebreak-style': ['error', 'unix'],              // Unix linebreaks
        'max-len': ['warn', {                              // Max line length
            'code': 120,
            'ignoreComments': true,
            'ignoreStrings': true,
            'ignoreTemplateLiterals': true,
            'ignoreUrls': true
        }],
        'new-cap': 'warn',                                 // Constructor names start with capital
        'new-parens': 'error',                             // Require parens for new
        'no-array-constructor': 'warn',                    // Disallow Array constructor
        'no-mixed-spaces-and-tabs': 'error',               // No mixing spaces and tabs
        'no-multiple-empty-lines': ['warn', {              // Max 2 empty lines
            'max': 2,
            'maxEOF': 1
        }],
        'no-trailing-spaces': 'warn',                      // No trailing whitespace
        'no-whitespace-before-property': 'warn',           // No whitespace before property
        'object-curly-spacing': ['warn', 'always'],        // Space in object literals
        'quotes': ['warn', 'single', {                     // Single quotes
            'avoidEscape': true,
            'allowTemplateLiterals': true
        }],
        'semi': ['error', 'always'],                       // Require semicolons
        'semi-spacing': 'warn',                            // Space after semicolon
        'space-before-blocks': 'warn',                     // Space before blocks
        'space-before-function-paren': ['warn', {          // Space before function paren
            'anonymous': 'never',
            'named': 'never',
            'asyncArrow': 'always'
        }],
        'space-in-parens': 'warn',                         // No space in parens
        'space-infix-ops': 'warn',                         // Space around operators
        'space-unary-ops': 'warn',                         // Space with unary operators
        'spaced-comment': 'warn',                          // Space after comment //
        
        // ========================================
        // ES6
        // ========================================
        
        'arrow-spacing': 'warn',                           // Space around arrow
        'no-duplicate-imports': 'error',                   // No duplicate imports
        'no-var': 'warn',                                  // Prefer const/let
        'prefer-const': 'warn',                            // Prefer const
        'prefer-arrow-callback': 'warn',                   // Prefer arrow functions
        'prefer-template': 'warn',                         // Prefer template literals
        'template-curly-spacing': 'warn'                   // No space in template literals
    },
    
    // ============================================
    // OVERRIDES
    // ============================================
    
    overrides: [
        {
            // Test files
            files: ['**/*.test.js', '**/*.spec.js', '**/tests/**/*.js'],
            env: {
                jest: true,
                mocha: true
            },
            rules: {
                'no-unused-expressions': 'off',
                'max-len': 'off'
            }
        },
        {
            // Config files
            files: ['*.config.js', '.*.js'],
            rules: {
                'no-console': 'off'
            }
        }
    ]
};
