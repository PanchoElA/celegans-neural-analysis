console.log('=== DEBUGGING NEURAL DASHBOARD ===');

// Test 1: Check if Plotly is loaded
console.log('Plotly loaded:', typeof Plotly !== 'undefined');

// Test 2: Check data variables
console.log('Data check:');
try {
    console.log('pcaScores length:', pcaScores ? pcaScores.length : 'undefined');
    console.log('timeData length:', timeData ? timeData.length : 'undefined');
    console.log('behaviorLabels length:', behaviorLabels ? behaviorLabels.length : 'undefined');
    
    // Test simple plot
    const testTrace = {
        x: [1, 2, 3, 4],
        y: [2, 4, 1, 3], 
        z: [1, 3, 2, 4],
        mode: 'markers+lines',
        type: 'scatter3d',
        name: 'Test'
    };
    
    const testLayout = {
        title: 'Debug Test Plot'
    };
    
    console.log('Creating test plot...');
    Plotly.newPlot('pca3d-plot', [testTrace], testLayout);
    console.log('Test plot created successfully!');
    
} catch (error) {
    console.error('Error in debugging:', error);
}
