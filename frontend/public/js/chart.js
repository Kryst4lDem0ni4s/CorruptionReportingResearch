/**
 * Corruption Reporting System - Chart Rendering
 * Version: 1.0.0
 * Description: D3.js-based chart and graph visualizations
 * 
 * This module provides:
 * - Network graph for coordination detection
 * - Bar charts for scores
 * - Progress bars
 * - Score gauges
 * - Responsive design
 * 
 * Dependencies: D3.js v7 (loaded via CDN in HTML)
 */

// ============================================
// CHART CONFIGURATION
// ============================================
const CHART_CONFIG = {
    colors: {
        high: '#28a745',
        medium: '#ffc107',
        low: '#dc3545',
        neutral: '#6c757d',
        primary: '#007bff'
    },
    transitions: {
        duration: 750,
        ease: 'cubicInOut'
    },
    margins: {
        top: 20,
        right: 20,
        bottom: 40,
        left: 50
    }
};

// ============================================
// NETWORK GRAPH
// ============================================

/**
 * Render coordination network graph
 * @param {string} containerId - Container element ID
 * @param {Object} data - Graph data with nodes and edges
 * @param {Object} options - Chart options
 */
function renderNetworkGraph(containerId, data, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container #${containerId} not found`);
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Check if D3 is available
    if (typeof d3 === 'undefined') {
        console.error('D3.js is not loaded');
        container.innerHTML = '<p class="text-danger">Chart library not loaded</p>';
        return;
    }
    
    // Parse data
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    
    if (nodes.length === 0) {
        container.innerHTML = '<p class="text-muted">No coordination detected</p>';
        return;
    }
    
    // Dimensions
    const width = options.width || container.clientWidth || 600;
    const height = options.height || 400;
    
    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height])
        .attr('style', 'max-width: 100%; height: auto;');
    
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(30));
    
    // Create links
    const link = svg.append('g')
        .selectAll('line')
        .data(edges)
        .join('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', d => Math.sqrt(d.weight || 1) * 2);
    
    // Create nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => d.size || 10)
        .attr('fill', d => getNodeColor(d.risk || 'low'))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .call(drag(simulation));
    
    // Add labels
    const label = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .join('text')
        .text(d => d.label || d.id)
        .attr('font-size', 12)
        .attr('dx', 15)
        .attr('dy', 4);
    
    // Add tooltips
    node.append('title')
        .text(d => `${d.label || d.id}\nRisk: ${d.risk || 'unknown'}`);
    
    // Update positions on tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    // Drag behavior
    function drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
}

/**
 * Get node color based on risk level
 * @param {string} risk - Risk level
 * @returns {string} Color code
 */
function getNodeColor(risk) {
    const colors = {
        high: CHART_CONFIG.colors.low,
        medium: CHART_CONFIG.colors.medium,
        low: CHART_CONFIG.colors.high,
        none: CHART_CONFIG.colors.neutral
    };
    return colors[risk] || colors.none;
}

// ============================================
// BAR CHART
// ============================================

/**
 * Render horizontal bar chart
 * @param {string} containerId - Container element ID
 * @param {Array} data - Array of {label, value, color}
 * @param {Object} options - Chart options
 */
function renderBarChart(containerId, data, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container #${containerId} not found`);
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Check if D3 is available
    if (typeof d3 === 'undefined') {
        console.error('D3.js is not loaded');
        return;
    }
    
    if (!data || data.length === 0) {
        container.innerHTML = '<p class="text-muted">No data available</p>';
        return;
    }
    
    // Dimensions
    const margin = CHART_CONFIG.margins;
    const width = (options.width || container.clientWidth || 600) - margin.left - margin.right;
    const height = (options.height || 300) - margin.top - margin.bottom;
    
    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', [0, 0, width + margin.left + margin.right, height + margin.top + margin.bottom])
        .attr('style', 'max-width: 100%; height: auto;')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const x = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.value)])
        .range([0, width]);
    
    const y = d3.scaleBand()
        .domain(data.map(d => d.label))
        .range([0, height])
        .padding(0.2);
    
    // Bars
    svg.selectAll('rect')
        .data(data)
        .join('rect')
        .attr('x', 0)
        .attr('y', d => y(d.label))
        .attr('width', 0)
        .attr('height', y.bandwidth())
        .attr('fill', d => d.color || CHART_CONFIG.colors.primary)
        .transition()
        .duration(CHART_CONFIG.transitions.duration)
        .attr('width', d => x(d.value));
    
    // Value labels
    svg.selectAll('text.value')
        .data(data)
        .join('text')
        .attr('class', 'value')
        .attr('x', d => x(d.value) + 5)
        .attr('y', d => y(d.label) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('font-size', 12)
        .text(d => d.value.toFixed(1) + (options.suffix || ''));
    
    // Y axis
    svg.append('g')
        .call(d3.axisLeft(y))
        .selectAll('text')
        .attr('font-size', 12);
    
    // X axis
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5))
        .selectAll('text')
        .attr('font-size', 12);
}

// ============================================
// SCORE GAUGE
// ============================================

/**
 * Render score gauge (semi-circle)
 * @param {string} containerId - Container element ID
 * @param {number} score - Score value (0-1)
 * @param {Object} options - Chart options
 */
function renderScoreGauge(containerId, score, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container #${containerId} not found`);
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Check if D3 is available
    if (typeof d3 === 'undefined') {
        console.error('D3.js is not loaded');
        return;
    }
    
    // Dimensions
    const width = options.width || container.clientWidth || 300;
    const height = (options.height || width / 2) + 40;
    const radius = Math.min(width, height * 2) / 2 - 20;
    
    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height])
        .attr('style', 'max-width: 100%; height: auto;');
    
    const g = svg.append('g')
        .attr('transform', `translate(${width / 2},${height - 20})`);
    
    // Arc generator
    const arc = d3.arc()
        .innerRadius(radius - 20)
        .outerRadius(radius)
        .startAngle(-Math.PI / 2);
    
    // Background arc
    g.append('path')
        .datum({ endAngle: Math.PI / 2 })
        .style('fill', '#e9ecef')
        .attr('d', arc);
    
    // Score arc
    const scoreColor = score >= 0.75 ? CHART_CONFIG.colors.high : 
                       score >= 0.5 ? CHART_CONFIG.colors.medium : 
                       CHART_CONFIG.colors.low;
    
    g.append('path')
        .datum({ endAngle: -Math.PI / 2 })
        .style('fill', scoreColor)
        .attr('d', arc)
        .transition()
        .duration(CHART_CONFIG.transitions.duration)
        .attrTween('d', function(d) {
            const interpolate = d3.interpolate(d.endAngle, -Math.PI / 2 + Math.PI * score);
            return function(t) {
                d.endAngle = interpolate(t);
                return arc(d);
            };
        });
    
    // Score text
    g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '-0.5em')
        .attr('font-size', 36)
        .attr('font-weight', 'bold')
        .attr('fill', scoreColor)
        .text((score * 100).toFixed(1) + '%');
    
    // Label
    if (options.label) {
        g.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '1.5em')
            .attr('font-size', 14)
            .attr('fill', '#6c757d')
            .text(options.label);
    }
}

// ============================================
// PROGRESS BAR
// ============================================

/**
 * Render animated progress bar
 * @param {string} containerId - Container element ID
 * @param {number} progress - Progress percentage (0-100)
 * @param {Object} options - Chart options
 */
function renderProgressBar(containerId, progress, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container #${containerId} not found`);
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Create progress bar HTML
    const progressBar = document.createElement('div');
    progressBar.className = 'progress';
    progressBar.style.height = options.height || '30px';
    
    const progressFill = document.createElement('div');
    progressFill.className = 'progress-bar progress-bar-striped progress-bar-animated';
    progressFill.setAttribute('role', 'progressbar');
    progressFill.setAttribute('aria-valuenow', progress);
    progressFill.setAttribute('aria-valuemin', '0');
    progressFill.setAttribute('aria-valuemax', '100');
    progressFill.style.width = '0%';
    progressFill.textContent = progress.toFixed(0) + '%';
    
    // Color based on progress
    if (progress >= 75) {
        progressFill.classList.add('bg-success');
    } else if (progress >= 50) {
        progressFill.classList.add('bg-warning');
    } else {
        progressFill.classList.add('bg-danger');
    }
    
    progressBar.appendChild(progressFill);
    container.appendChild(progressBar);
    
    // Animate
    setTimeout(() => {
        progressFill.style.transition = 'width 750ms ease-in-out';
        progressFill.style.width = progress + '%';
    }, 100);
}

// ============================================
// LINE CHART
// ============================================

/**
 * Render line chart
 * @param {string} containerId - Container element ID
 * @param {Array} data - Array of {x, y} points
 * @param {Object} options - Chart options
 */
function renderLineChart(containerId, data, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container #${containerId} not found`);
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Check if D3 is available
    if (typeof d3 === 'undefined') {
        console.error('D3.js is not loaded');
        return;
    }
    
    if (!data || data.length === 0) {
        container.innerHTML = '<p class="text-muted">No data available</p>';
        return;
    }
    
    // Dimensions
    const margin = CHART_CONFIG.margins;
    const width = (options.width || container.clientWidth || 600) - margin.left - margin.right;
    const height = (options.height || 300) - margin.top - margin.bottom;
    
    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', [0, 0, width + margin.left + margin.right, height + margin.top + margin.bottom])
        .attr('style', 'max-width: 100%; height: auto;')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const x = d3.scaleLinear()
        .domain(d3.extent(data, d => d.x))
        .range([0, width]);
    
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.y)])
        .range([height, 0]);
    
    // Line generator
    const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y))
        .curve(d3.curveMonotoneX);
    
    // Draw line
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', options.color || CHART_CONFIG.colors.primary)
        .attr('stroke-width', 2)
        .attr('d', line);
    
    // Add dots
    svg.selectAll('circle')
        .data(data)
        .join('circle')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('r', 4)
        .attr('fill', options.color || CHART_CONFIG.colors.primary);
    
    // X axis
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5))
        .selectAll('text')
        .attr('font-size', 12);
    
    // Y axis
    svg.append('g')
        .call(d3.axisLeft(y).ticks(5))
        .selectAll('text')
        .attr('font-size', 12);
    
    // Axis labels
    if (options.xLabel) {
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('x', width / 2)
            .attr('y', height + 35)
            .attr('font-size', 12)
            .text(options.xLabel);
    }
    
    if (options.yLabel) {
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -35)
            .attr('font-size', 12)
            .text(options.yLabel);
    }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Resize all charts in container
 * @param {string} containerId - Container element ID
 */
function resizeCharts(containerId) {
    // This would re-render charts on window resize
    // Implementation depends on tracking active charts
    console.log('Resize charts in:', containerId);
}

/**
 * Clear all charts
 */
function clearAllCharts() {
    // Remove all SVG elements
    document.querySelectorAll('svg').forEach(svg => {
        if (svg.parentElement) {
            svg.remove();
        }
    });
}

// ============================================
// EXPORTS
// ============================================

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ChartRenderer = {
        renderNetworkGraph,
        renderBarChart,
        renderScoreGauge,
        renderProgressBar,
        renderLineChart,
        resizeCharts,
        clearAllCharts,
        config: CHART_CONFIG
    };
}

// Auto-resize on window resize
if (typeof window !== 'undefined') {
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            // Could re-render responsive charts here
            console.log('Window resized - charts may need re-rendering');
        }, 250);
    });
}
